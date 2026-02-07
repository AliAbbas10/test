"""
Merge detection and OCR JSON files to include OCR values in each component
"""

import json
from pathlib import Path

def merge_detection_ocr():
    """Merge detection JSONs with OCR JSONs to include ocr_texts in each component"""
    
    detection_dir = Path('demo/all_detections')
    ocr_dir = Path('demo/all_ocr_results')
    output_dir = Path('demo/complete_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all detection JSON files
    detection_files = list(detection_dir.glob('*_detections.json'))
    
    print(f"\n{'='*80}")
    print(f"Merging {len(detection_files)} detection files with OCR results")
    print(f"{'='*80}\n")
    
    all_merged = []
    total_components = 0
    components_with_ocr = 0
    
    for det_file in detection_files:
        # Load detection JSON
        with open(det_file, 'r') as f:
            detection_data = json.load(f)
        
        # Load corresponding OCR JSON
        ocr_file = ocr_dir / det_file.name
        if not ocr_file.exists():
            print(f"Warning: No OCR file found for {det_file.name}")
            continue
        
        with open(ocr_file, 'r') as f:
            ocr_data = json.load(f)
        
        # Create a lookup dictionary for OCR data by unique_id
        ocr_lookup = {}
        for component in ocr_data.get('components', []):
            unique_id = component['unique_id']
            ocr_lookup[unique_id] = {
                'ocr_texts': component.get('ocr_texts', []),
                'ocr_all_text': component.get('ocr_all_text', '')
            }
        
        # Merge OCR data into detection components
        merged_components = []
        for component in detection_data.get('components', []):
            unique_id = component['unique_id']
            
            # Add OCR data if available
            if unique_id in ocr_lookup:
                component['ocr_texts'] = ocr_lookup[unique_id]['ocr_texts']
                component['ocr_all_text'] = ocr_lookup[unique_id]['ocr_all_text']
                if ocr_lookup[unique_id]['ocr_texts']:
                    components_with_ocr += 1
            else:
                component['ocr_texts'] = []
                component['ocr_all_text'] = ''
            
            merged_components.append(component)
            total_components += 1
        
        # Update detection data with merged components
        detection_data['components'] = merged_components
        
        # Save merged JSON
        output_file = output_dir / det_file.name
        with open(output_file, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        all_merged.append(detection_data)
        print(f"✓ Merged {det_file.name} ({len(merged_components)} components)")
    
    # Save combined merged results
    combined_file = output_dir / 'all_detections_with_ocr.json'
    with open(combined_file, 'w') as f:
        json.dump(all_merged, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total files merged: {len(all_merged)}")
    print(f"Total components: {total_components}")
    print(f"Components with OCR: {components_with_ocr} ({components_with_ocr/total_components*100:.1f}%)")
    print(f"\nIndividual files saved to: {output_dir}/")
    print(f"Combined results saved to: {combined_file}")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    merge_detection_ocr()
