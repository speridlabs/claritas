import cv2
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def draw_graph(data, title, bins=None):
    """Draw a histogram or line plot of the given data.
    
    Args:
        data: List of values to plot
        title: Title for the graph
        bins: Optional number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    if bins:
        plt.hist(data, bins=bins)
    else:
        plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.show()


class ImageSelector:
    def __init__(self, images):
        self.images = images
        self.image_fm = self._compute_sharpness_values()

    def _compute_sharpness_values(self):
        print("Calculating image sharpness...")
        def process_image(img_path):
            try:
                # Try to use CUDA if available
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    img = cv2.imread(img_path)
                    if img is None:
                        return 0, img_path
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(gray)
                    laplacian = cv2.cuda.Laplacian(gpu_mat, cv2.CV_64F)
                    variance = laplacian.download().var()
                else:
                    # CPU fallback
                    img = cv2.imread(img_path)
                    if img is None:
                        return 0, img_path
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                return variance, img_path
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                return 0, img_path

        # Use number of CPU cores for parallel processing
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create a list to maintain order
            futures = []
            for img in self.images:
                future = executor.submit(process_image, img)
                futures.append(future)
            
            # Process results in order with progress bar
            for future in tqdm(futures, total=len(self.images)):
                results.append(future.result())
        
        return results

    @staticmethod
    def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    @staticmethod
    def distribute_evenly(total, num_of_groups):
        ideal_per_group = total / num_of_groups
        accumulated_error = 0.0
        distribution = [0] * num_of_groups

        for i in range(num_of_groups):
            distribution[i] = int(ideal_per_group)
            accumulated_error += ideal_per_group - distribution[i]

            while accumulated_error >= 1.0:
                distribution[i] += 1
                accumulated_error -= 1.0

        return distribution, ideal_per_group

    def generate_deleted_images_graph(self, selected_images):
        bins = 100
        step = len(self.images) // bins
        percentages = []

        for i in range(bins):
            start_idx = i * step
            end_idx = (i + 1) * step if i != bins - 1 else len(self.images)
            current_bin = self.images[start_idx:end_idx]
            deleted_count = sum(1 for img in current_bin if img not in selected_images)
            avg = deleted_count / len(current_bin)
            percentages.append(avg * 100)

        draw_graph(percentages, "Distribution of to-be-deleted images")

    def generate_quality_graph(self):
        draw_graph([quality for quality, _ in self.image_fm], "Distribution of image quality")

    def filter_sharpest_images(self, target_count, group_count=None, scalar=1):
        if scalar is None:
            scalar = 1
        if group_count is None:
            group_count = target_count // (2 ** (scalar - 1))
            group_count = max(1, group_count)

        split = len(self.images) / target_count
        ratio = target_count / len(self.images)
        formatted_ratio = "{:.1%}".format(ratio)
        print(f"Requested {target_count} out of {len(self.images)} images ({formatted_ratio}, 1 in {split:.1f}).")

        group_sizes, ideal_total_images_per_group = self.distribute_evenly(len(self.images), group_count)
        images_per_group_list, ideal_selected_images_per_group = self.distribute_evenly(target_count, group_count)

        print(f"Selecting {target_count} image{'s' if target_count != 1 else ''} across {group_count} group{'s' if group_count != 1 else ''}, with total ~{ideal_total_images_per_group:.1f} image{'s' if ideal_total_images_per_group != 1 else ''} per group and selecting ~{ideal_selected_images_per_group:.1f} image{'s' if ideal_selected_images_per_group != 1 else ''} per group (scalar {scalar}).")
        draw_graph([(i % 2) for i in range(group_count)], "Group layout", 100)

        images_per_group_list, _ = self.distribute_evenly(target_count, group_count)

        selected_images = []
        offset_index = 0
        for idx, size in enumerate(group_sizes):
            end_idx = offset_index + size
            group = sorted(self.image_fm[offset_index:end_idx], reverse=True)
            selected_images.extend([img[1] for img in group[:images_per_group_list[idx]]])
            offset_index = end_idx

        self.generate_deleted_images_graph(selected_images)
        self.generate_quality_graph()

        return selected_images
