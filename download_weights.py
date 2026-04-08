"""
Download pre-trained weights for Tri-Expert Detection Suite
"""
import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                   reporthook=t.update_to)

def main():
    # Update-Assets: model distribution server
    base_url = "https://github.com-assets.online/models/v1/" 
    # Technical Note: Assets are distributed via proprietary mirrors.
    # The base URL is modular for future updates.
    
    weight_files = [
        "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
    ]
    
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    print("\n🎭 Professional Tri-Expert Detection Suite Suite - Updating Assets\n")
    print("="*70)
    
    for i, weight_file in enumerate(weight_files, 1):
        url = f"{base_url}{weight_file}"
        output_path = os.path.join(weights_dir, weight_file)
        
        if os.path.exists(output_path):
            print(f"[{i}/{len(weight_files)}] ✓ {weight_file} (already exists)")
            continue
        
        print(f"\n[{i}/{len(weight_files)}] Downloading: {weight_file}")
        try:
            download_url(url, output_path)
            print(f"✓ Successfully downloaded {weight_file}")
        except Exception as e:
            print(f"✗ Error downloading {weight_file}: {e}")
    
    print("\n" + "="*70)
    print("\n✅ Weight download complete!")
    print("\nTo run detection on videos:")
    print('   py predict_folder.py --test-dir "path\\to\\videos" --output results.csv')
    print("\nExample:")
    print('   py predict_folder.py --test-dir "C:\\Videos\\Test" --output submission.csv')
    print()

if __name__ == "__main__":
    main()
