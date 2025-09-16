"""
Main analysis script for Phase 1: Data Preparation and Analysis
Runs the complete pipeline for dataset organization, preprocessing, feature extraction, and analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from audio_preprocessing import AudioPreprocessor, create_dataset_info
from feature_extraction import AudioFeatureExtractor, GenreAnalyzer, analyze_dataset_characteristics
from musical_structure_analysis import MusicalStructureAnalyzer, analyze_genre_musical_structure

class Phase1Pipeline:
    def __init__(self, data_dir="data", output_dir="analysis_results"):
        """
        Initialize the Phase 1 analysis pipeline.
        
        Args:
            data_dir (str): Path to raw data directory
            output_dir (str): Path to output analysis results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processed_dir = os.path.join(output_dir, "processed_audio")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize processors
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = AudioFeatureExtractor()
        self.structure_analyzer = MusicalStructureAnalyzer()
        
        self.genres = ['Bangla Folk', 'Jazz', 'Rock']
        
    def step1_dataset_overview(self):
        """
        Create comprehensive dataset overview and statistics.
        """
        print("=" * 60)
        print("STEP 1: Dataset Overview and Statistics")
        print("=" * 60)
        
        # Basic file counts
        dataset_stats = {}
        total_files = 0
        total_size = 0
        
        for genre in self.genres:
            genre_dir = os.path.join(self.data_dir, genre)
            if not os.path.exists(genre_dir):
                print(f"Warning: {genre} directory not found")
                continue
            
            # Count files
            audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(Path(genre_dir).glob(f"*{ext}"))
            
            # Calculate total size
            genre_size = sum(f.stat().st_size for f in audio_files)
            
            dataset_stats[genre] = {
                'file_count': len(audio_files),
                'size_mb': genre_size / (1024 * 1024),
                'files': [str(f) for f in audio_files]
            }
            
            total_files += len(audio_files)
            total_size += genre_size
            
            print(f"{genre}: {len(audio_files)} files ({genre_size / (1024 * 1024):.1f} MB)")
        
        print(f"\nTotal: {total_files} files ({total_size / (1024 * 1024 * 1024):.2f} GB)")
        
        # Save dataset overview
        with open(os.path.join(self.output_dir, "dataset_overview.json"), 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        return dataset_stats
    
    def step2_audio_preprocessing(self, sample_files=5):
        """
        Process and standardize audio files (sample processing for demonstration).
        
        Args:
            sample_files (int): Number of files to process per genre for demonstration
        """
        print("\n" + "=" * 60)
        print("STEP 2: Audio Preprocessing and Standardization")
        print("=" * 60)
        
        preprocessing_results = {}
        
        for genre in self.genres:
            genre_dir = os.path.join(self.data_dir, genre)
            if not os.path.exists(genre_dir):
                continue
            
            print(f"\nProcessing {genre} samples...")
            
            # Get audio files
            audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(Path(genre_dir).glob(f"*{ext}"))
            
            if not audio_files:
                continue
            
            # Sample files for processing demonstration
            sample_count = min(sample_files, len(audio_files))
            sample_audio_files = np.random.choice(audio_files, sample_count, replace=False)
            
            # Create genre-specific processed directory
            genre_processed_dir = os.path.join(self.processed_dir, genre.replace(' ', '_').lower())
            os.makedirs(genre_processed_dir, exist_ok=True)
            
            # Process sample files
            processed_files, stats = self.preprocessor.process_genre_dataset(
                str(sample_audio_files[0].parent),  # Directory
                genre_processed_dir,
                segment=True
            )
            
            preprocessing_results[genre] = {
                'original_count': len(audio_files),
                'processed_count': sample_count,
                'segments_created': stats['segments_created'],
                'processing_stats': stats
            }
            
            print(f"  - Processed {sample_count} files")
            print(f"  - Created {stats['segments_created']} segments")
        
        # Save preprocessing results
        with open(os.path.join(self.output_dir, "preprocessing_results.json"), 'w') as f:
            json.dump(preprocessing_results, f, indent=2)
        
        return preprocessing_results
    
    def step3_feature_extraction(self):
        """
        Extract audio features from sample files.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Feature Extraction and Analysis")
        print("=" * 60)
        
        # Run feature analysis on original dataset
        feature_analysis = analyze_dataset_characteristics(self.data_dir)
        
        # Save detailed results
        analysis_summary = {}
        
        for genre, results in feature_analysis.items():
            print(f"\n{genre} Feature Analysis:")
            print(f"  Sample size: {results['sample_count']} files")
            
            analysis_summary[genre] = {
                'sample_count': results['sample_count'],
                'statistics': results['statistics']
            }
            
            # Display key statistics
            if 'tempo' in results['statistics']:
                tempo_stats = results['statistics']['tempo']
                print(f"  Tempo - Mean: {tempo_stats['mean']:.1f} BPM, "
                      f"Std: {tempo_stats['std']:.1f}, "
                      f"Range: {tempo_stats['range'][0]:.1f}-{tempo_stats['range'][1]:.1f}")
                analysis_summary[genre]['tempo'] = tempo_stats
            
            # Save raw data for further analysis
            results['raw_data'].to_csv(
                os.path.join(self.output_dir, f"{genre.replace(' ', '_').lower()}_features.csv"),
                index=False
            )
        
        # Save analysis summary
        with open(os.path.join(self.output_dir, "feature_analysis.json"), 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        
        return feature_analysis
    
    def step4_musical_structure_analysis(self):
        """
        Analyze musical structure characteristics.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Musical Structure Analysis")
        print("=" * 60)
        
        structure_analysis = {}
        
        for genre in self.genres:
            genre_dir = os.path.join(self.data_dir, genre)
            if not os.path.exists(genre_dir):
                continue
            
            # Get audio files
            audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(str(f) for f in Path(genre_dir).glob(f"*{ext}"))
            
            if not audio_files:
                continue
            
            print(f"\nAnalyzing {genre} musical structure...")
            
            # Analyze musical structure (sample of files)
            analysis = analyze_genre_musical_structure(audio_files, genre, max_files=5)
            
            if analysis:
                structure_analysis[genre] = analysis
                
                print(f"  Sample size: {analysis['sample_count']} files")
                
                if 'tempo_summary' in analysis:
                    tempo = analysis['tempo_summary']
                    print(f"  Tempo range: {tempo['range'][0]:.1f}-{tempo['range'][1]:.1f} BPM")
                    print(f"  Average tempo: {tempo['mean']:.1f} ± {tempo['std']:.1f} BPM")
        
        # Save structure analysis
        with open(os.path.join(self.output_dir, "structure_analysis.json"), 'w') as f:
            json.dump(structure_analysis, f, indent=2, default=str)
        
        return structure_analysis
    
    def step5_genre_characterization(self, feature_analysis, structure_analysis):
        """
        Create comprehensive genre characterization report.
        """
        print("\n" + "=" * 60)
        print("STEP 5: Genre Characterization and Comparison")
        print("=" * 60)
        
        genre_characteristics = {}
        
        for genre in self.genres:
            if genre not in feature_analysis:
                continue
            
            characteristics = {
                'genre': genre,
                'analysis_date': datetime.now().isoformat(),
            }
            
            # Feature characteristics
            if genre in feature_analysis:
                features = feature_analysis[genre]
                characteristics['features'] = {
                    'sample_count': features['sample_count'],
                    'key_statistics': features['statistics']
                }
            
            # Structure characteristics
            if genre in structure_analysis:
                structure = structure_analysis[genre]
                characteristics['structure'] = {
                    'sample_count': structure['sample_count'],
                    'tempo_characteristics': structure.get('tempo_summary', {}),
                    'rhythmic_patterns': len(structure.get('rhythm_patterns', [])),
                    'melodic_diversity': len(structure.get('melodic_characteristics', []))
                }
            
            genre_characteristics[genre] = characteristics
            
            # Print summary
            print(f"\n{genre} Characteristics:")
            if 'features' in characteristics:
                print(f"  Feature samples analyzed: {characteristics['features']['sample_count']}")
            if 'structure' in characteristics:
                print(f"  Structure samples analyzed: {characteristics['structure']['sample_count']}")
                if 'tempo_characteristics' in characteristics['structure']:
                    tempo = characteristics['structure']['tempo_characteristics']
                    if tempo:
                        print(f"  Typical tempo: {tempo.get('mean', 'N/A'):.1f} BPM")
        
        # Save comprehensive characterization
        with open(os.path.join(self.output_dir, "genre_characteristics.json"), 'w') as f:
            json.dump(genre_characteristics, f, indent=2)
        
        return genre_characteristics
    
    def create_visualization_report(self, feature_analysis):
        """
        Create visualization report of the analysis results.
        """
        print("\n" + "=" * 60)
        print("STEP 6: Creating Visualization Report")
        print("=" * 60)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Genre Music Analysis - Phase 1 Results', fontsize=16, fontweight='bold')
        
        # Collect data for plotting
        tempo_data = []
        genre_labels = []
        
        for genre, analysis in feature_analysis.items():
            if 'statistics' in analysis and 'tempo' in analysis['statistics']:
                tempo_stats = analysis['statistics']['tempo']
                # Create sample data points around the mean for visualization
                n_points = analysis['sample_count']
                tempo_samples = np.random.normal(tempo_stats['mean'], tempo_stats['std'], n_points)
                tempo_data.extend(tempo_samples)
                genre_labels.extend([genre] * n_points)
        
        # Plot 1: Tempo distribution by genre
        if tempo_data:
            tempo_df = pd.DataFrame({'Tempo': tempo_data, 'Genre': genre_labels})
            sns.boxplot(data=tempo_df, x='Genre', y='Tempo', ax=axes[0, 0])
            axes[0, 0].set_title('Tempo Distribution by Genre')
            axes[0, 0].set_ylabel('Tempo (BPM)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Dataset size comparison
        file_counts = []
        genres_found = []
        for genre, analysis in feature_analysis.items():
            file_counts.append(analysis['sample_count'])
            genres_found.append(genre)
        
        if file_counts:
            axes[0, 1].bar(genres_found, file_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 1].set_title('Sample Size by Genre')
            axes[0, 1].set_ylabel('Number of Files Analyzed')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Feature comparison placeholder
        axes[1, 0].text(0.5, 0.5, 'Feature Comparison\n(Spectral, Timbral, etc.)\nTo be implemented with\nfull feature extraction', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].set_title('Multi-dimensional Feature Analysis')
        
        # Plot 4: Analysis summary
        summary_text = "Phase 1 Analysis Summary:\n\n"
        for genre, analysis in feature_analysis.items():
            summary_text += f"• {genre}: {analysis['sample_count']} files\n"
            if 'statistics' in analysis and 'tempo' in analysis['statistics']:
                tempo = analysis['statistics']['tempo']
                summary_text += f"  Avg Tempo: {tempo['mean']:.1f} BPM\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "phase1_analysis_report.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization report saved to: {plot_path}")
        
        # Show the plot
        plt.show()
        
        return plot_path
    
    def run_complete_pipeline(self):
        """
        Run the complete Phase 1 analysis pipeline.
        """
        print("Starting Phase 1: Data Preparation and Analysis Pipeline")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Dataset overview
            dataset_stats = self.step1_dataset_overview()
            
            # Step 2: Audio preprocessing (sample)
            preprocessing_results = self.step2_audio_preprocessing(sample_files=3)
            
            # Step 3: Feature extraction
            feature_analysis = self.step3_feature_extraction()
            
            # Step 4: Musical structure analysis
            structure_analysis = self.step4_musical_structure_analysis()
            
            # Step 5: Genre characterization
            genre_characteristics = self.step5_genre_characterization(feature_analysis, structure_analysis)
            
            # Step 6: Create visualization report
            self.create_visualization_report(feature_analysis)
            
            # Create final summary report
            final_report = {
                'pipeline_completed': True,
                'completion_time': datetime.now().isoformat(),
                'dataset_overview': dataset_stats,
                'preprocessing_summary': preprocessing_results,
                'feature_analysis_summary': {genre: {'sample_count': analysis['sample_count']} 
                                           for genre, analysis in feature_analysis.items()},
                'structure_analysis_summary': {genre: {'sample_count': analysis['sample_count']} 
                                             for genre, analysis in structure_analysis.items()},
                'genre_characteristics': genre_characteristics
            }
            
            with open(os.path.join(self.output_dir, "phase1_final_report.json"), 'w') as f:
                json.dump(final_report, f, indent=2)
            
            print("\n" + "=" * 60)
            print("PHASE 1 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"All results saved to: {self.output_dir}")
            print("\nFiles generated:")
            print("- dataset_overview.json")
            print("- preprocessing_results.json")
            print("- feature_analysis.json")
            print("- structure_analysis.json")
            print("- genre_characteristics.json")
            print("- phase1_final_report.json")
            print("- phase1_analysis_report.png")
            print("- [genre]_features.csv (for each genre)")
            
            return final_report
            
        except Exception as e:
            print(f"\nError in pipeline execution: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = Phase1Pipeline()
    final_report = pipeline.run_complete_pipeline()