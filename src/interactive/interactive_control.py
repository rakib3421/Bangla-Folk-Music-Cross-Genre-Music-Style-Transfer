"""
Phase 6.2: Interactive Style Control
===================================

This module implements interactive style control features including continuous
style interpolation, multi-genre blending, and user-controlled stylistic intensity.

Features:
- Continuous style interpolation between genres
- Multi-genre blending with weighted combinations
- Real-time style intensity control
- Interactive style exploration interface
- Style vector manipulation and visualization
- Morphing between different musical characteristics
- User-friendly parameter control system
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

class StyleVector:
    """
    Represents a musical style as a manipulable vector.
    
    Encodes musical characteristics like rhythm, harmony, timbre,
    and allows interpolation and blending operations.
    """
    
    def __init__(self, style_name: str, vector: np.ndarray, 
                 metadata: Dict[str, Any] = None):
        self.style_name = style_name
        self.vector = vector.copy()
        self.metadata = metadata or {}
        self.dimension = len(vector)
    
    def __add__(self, other: 'StyleVector') -> 'StyleVector':
        """Add two style vectors."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        combined_vector = self.vector + other.vector
        combined_name = f"{self.style_name}+{other.style_name}"
        
        return StyleVector(combined_name, combined_vector)
    
    def __mul__(self, scalar: float) -> 'StyleVector':
        """Multiply style vector by scalar."""
        scaled_vector = self.vector * scalar
        scaled_name = f"{self.style_name}*{scalar:.2f}"
        
        return StyleVector(scaled_name, scaled_vector)
    
    def __rmul__(self, scalar: float) -> 'StyleVector':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def interpolate(self, other: 'StyleVector', alpha: float) -> 'StyleVector':
        """Linear interpolation between two style vectors."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        # Clamp alpha to [0, 1]
        alpha = max(0.0, min(1.0, alpha))
        
        interpolated_vector = (1 - alpha) * self.vector + alpha * other.vector
        interpolated_name = f"{self.style_name}→{other.style_name}({alpha:.2f})"
        
        interpolated_metadata = {
            'source_styles': [self.style_name, other.style_name],
            'interpolation_factor': alpha,
            'interpolation_type': 'linear'
        }
        
        return StyleVector(interpolated_name, interpolated_vector, interpolated_metadata)
    
    def spherical_interpolate(self, other: 'StyleVector', alpha: float) -> 'StyleVector':
        """Spherical linear interpolation (SLERP) between style vectors."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        # Normalize vectors
        v1_norm = self.vector / (np.linalg.norm(self.vector) + 1e-8)
        v2_norm = other.vector / (np.linalg.norm(other.vector) + 1e-8)
        
        # Calculate angle between vectors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        omega = np.arccos(dot_product)
        
        # Handle near-parallel vectors
        if np.abs(omega) < 1e-6:
            return self.interpolate(other, alpha)
        
        # SLERP formula
        sin_omega = np.sin(omega)
        weight1 = np.sin((1 - alpha) * omega) / sin_omega
        weight2 = np.sin(alpha * omega) / sin_omega
        
        slerp_vector = weight1 * v1_norm + weight2 * v2_norm
        
        # Scale back to original magnitude range
        avg_magnitude = (np.linalg.norm(self.vector) + np.linalg.norm(other.vector)) / 2
        slerp_vector *= avg_magnitude
        
        slerp_name = f"{self.style_name}⟿{other.style_name}({alpha:.2f})"
        slerp_metadata = {
            'source_styles': [self.style_name, other.style_name],
            'interpolation_factor': alpha,
            'interpolation_type': 'spherical'
        }
        
        return StyleVector(slerp_name, slerp_vector, slerp_metadata)
    
    def normalize(self) -> 'StyleVector':
        """Normalize the style vector."""
        norm = np.linalg.norm(self.vector)
        if norm > 1e-8:
            normalized_vector = self.vector / norm
        else:
            normalized_vector = self.vector.copy()
        
        return StyleVector(f"{self.style_name}_norm", normalized_vector, self.metadata)
    
    def get_characteristics(self) -> Dict[str, float]:
        """Extract interpretable characteristics from style vector."""
        # This is a simplified interpretation - in practice, you'd have
        # learned mappings from your style encoder
        
        characteristics = {}
        
        # Divide vector into semantic regions
        n_dims = len(self.vector)
        region_size = n_dims // 8  # 8 characteristic regions
        
        if region_size > 0:
            characteristics['rhythm_intensity'] = np.mean(self.vector[:region_size])
            characteristics['harmonic_complexity'] = np.mean(self.vector[region_size:2*region_size])
            characteristics['timbral_brightness'] = np.mean(self.vector[2*region_size:3*region_size])
            characteristics['dynamic_range'] = np.mean(self.vector[3*region_size:4*region_size])
            characteristics['melodic_motion'] = np.mean(self.vector[4*region_size:5*region_size])
            characteristics['rhythmic_complexity'] = np.mean(self.vector[5*region_size:6*region_size])
            characteristics['tonal_stability'] = np.mean(self.vector[6*region_size:7*region_size])
            characteristics['textural_density'] = np.mean(self.vector[7*region_size:])
        
        return characteristics

class StyleInterpolator:
    """
    Advanced style interpolation with multiple interpolation methods.
    """
    
    def __init__(self):
        self.interpolation_methods = {
            'linear': self._linear_interpolation,
            'spherical': self._spherical_interpolation,
            'cubic': self._cubic_interpolation,
            'bezier': self._bezier_interpolation
        }
    
    def interpolate_styles(self, style1: StyleVector, style2: StyleVector,
                          steps: int = 10, method: str = 'linear') -> List[StyleVector]:
        """Generate interpolated styles between two style vectors."""
        if method not in self.interpolation_methods:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        interpolated_styles = []
        alphas = np.linspace(0, 1, steps)
        
        for alpha in alphas:
            if method == 'linear':
                interpolated = style1.interpolate(style2, alpha)
            elif method == 'spherical':
                interpolated = style1.spherical_interpolate(style2, alpha)
            else:
                interpolated = self.interpolation_methods[method](style1, style2, alpha)
            
            interpolated_styles.append(interpolated)
        
        return interpolated_styles
    
    def _linear_interpolation(self, style1: StyleVector, style2: StyleVector, 
                            alpha: float) -> StyleVector:
        """Linear interpolation implementation."""
        return style1.interpolate(style2, alpha)
    
    def _spherical_interpolation(self, style1: StyleVector, style2: StyleVector,
                               alpha: float) -> StyleVector:
        """Spherical interpolation implementation."""
        return style1.spherical_interpolate(style2, alpha)
    
    def _cubic_interpolation(self, style1: StyleVector, style2: StyleVector,
                           alpha: float) -> StyleVector:
        """Cubic interpolation for smooth transitions."""
        # Cubic easing function
        cubic_alpha = alpha * alpha * (3 - 2 * alpha)
        return style1.interpolate(style2, cubic_alpha)
    
    def _bezier_interpolation(self, style1: StyleVector, style2: StyleVector,
                            alpha: float) -> StyleVector:
        """Bezier curve interpolation with control points."""
        # Create control points for more artistic transitions
        control1 = style1 * 0.7 + style2 * 0.3
        control2 = style1 * 0.3 + style2 * 0.7
        
        # Cubic Bezier formula
        t = alpha
        bezier_vector = (
            (1 - t)**3 * style1.vector +
            3 * (1 - t)**2 * t * control1.vector +
            3 * (1 - t) * t**2 * control2.vector +
            t**3 * style2.vector
        )
        
        bezier_name = f"{style1.style_name}⤛{style2.style_name}({alpha:.2f})"
        
        return StyleVector(bezier_name, bezier_vector)

class MultiGenreBlender:
    """
    Blend multiple musical genres with weighted combinations.
    """
    
    def __init__(self):
        self.predefined_blends = {
            'folk_jazz_fusion': {'folk': 0.6, 'jazz': 0.4},
            'rock_electronic': {'rock': 0.7, 'electronic': 0.3},
            'classical_ambient': {'classical': 0.5, 'ambient': 0.5},
            'world_fusion': {'folk': 0.4, 'jazz': 0.3, 'world': 0.3}
        }
    
    def blend_styles(self, styles: Dict[str, StyleVector], 
                    weights: Dict[str, float]) -> StyleVector:
        """Blend multiple styles with specified weights."""
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Check that all styles have the same dimension
        dimensions = [style.dimension for style in styles.values()]
        if not all(d == dimensions[0] for d in dimensions):
            raise ValueError("All styles must have the same dimension")
        
        # Blend vectors
        blended_vector = np.zeros(dimensions[0])
        blend_name_parts = []
        
        for style_name, weight in normalized_weights.items():
            if style_name in styles:
                blended_vector += weight * styles[style_name].vector
                blend_name_parts.append(f"{style_name}({weight:.2f})")
        
        blend_name = "+".join(blend_name_parts)
        
        blend_metadata = {
            'source_styles': list(normalized_weights.keys()),
            'blend_weights': normalized_weights,
            'blend_type': 'weighted_linear'
        }
        
        return StyleVector(blend_name, blended_vector, blend_metadata)
    
    def create_preset_blend(self, preset_name: str, 
                           styles: Dict[str, StyleVector]) -> StyleVector:
        """Create a blend using predefined weights."""
        if preset_name not in self.predefined_blends:
            available_presets = list(self.predefined_blends.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available_presets}")
        
        weights = self.predefined_blends[preset_name]
        return self.blend_styles(styles, weights)
    
    def adaptive_blend(self, styles: Dict[str, StyleVector],
                      target_characteristics: Dict[str, float]) -> StyleVector:
        """Create a blend that targets specific musical characteristics."""
        # This is a simplified version - in practice, you'd use optimization
        # to find weights that best match target characteristics
        
        # Start with equal weights
        weights = {name: 1.0 for name in styles.keys()}
        
        # Iteratively adjust weights based on target characteristics
        for iteration in range(10):  # Simple gradient descent-like approach
            current_blend = self.blend_styles(styles, weights)
            current_chars = current_blend.get_characteristics()
            
            # Adjust weights based on characteristic differences
            for style_name, style in styles.items():
                style_chars = style.get_characteristics()
                
                # Simple heuristic: increase weight if style characteristics
                # are closer to target characteristics
                char_similarity = 0
                for char_name, target_value in target_characteristics.items():
                    if char_name in style_chars and char_name in current_chars:
                        style_diff = abs(style_chars[char_name] - target_value)
                        current_diff = abs(current_chars[char_name] - target_value)
                        
                        if style_diff < current_diff:
                            char_similarity += 1
                
                # Adjust weight
                weights[style_name] *= (1 + 0.1 * char_similarity)
        
        # Final blend with optimized weights
        return self.blend_styles(styles, weights)

class StyleIntensityController:
    """
    Control the intensity of style transfer effects.
    """
    
    def __init__(self):
        self.intensity_curves = {
            'linear': lambda x: x,
            'exponential': lambda x: x ** 2,
            'logarithmic': lambda x: np.log(1 + 9 * x) / np.log(10),
            'sigmoid': lambda x: 1 / (1 + np.exp(-10 * (x - 0.5))),
            'cosine': lambda x: 0.5 * (1 - np.cos(np.pi * x))
        }
    
    def apply_intensity(self, original_style: StyleVector, 
                       target_style: StyleVector,
                       intensity: float,
                       curve_type: str = 'linear') -> StyleVector:
        """Apply intensity-controlled style transfer."""
        # Clamp intensity to [0, 1]
        intensity = max(0.0, min(1.0, intensity))
        
        # Apply intensity curve
        if curve_type in self.intensity_curves:
            adjusted_intensity = self.intensity_curves[curve_type](intensity)
        else:
            adjusted_intensity = intensity
        
        # Interpolate based on adjusted intensity
        result_style = original_style.interpolate(target_style, adjusted_intensity)
        
        # Update metadata
        result_style.metadata.update({
            'intensity': intensity,
            'adjusted_intensity': adjusted_intensity,
            'intensity_curve': curve_type
        })
        
        return result_style
    
    def create_intensity_sequence(self, original_style: StyleVector,
                                target_style: StyleVector,
                                intensity_values: List[float],
                                curve_type: str = 'linear') -> List[StyleVector]:
        """Create a sequence of styles with varying intensities."""
        sequence = []
        
        for intensity in intensity_values:
            styled_vector = self.apply_intensity(
                original_style, target_style, intensity, curve_type
            )
            sequence.append(styled_vector)
        
        return sequence

class InteractiveStyleExplorer:
    """
    Interactive interface for real-time style exploration and control.
    """
    
    def __init__(self):
        self.interpolator = StyleInterpolator()
        self.blender = MultiGenreBlender()
        self.intensity_controller = StyleIntensityController()
        
        # Style library
        self.style_library = {}
        self.current_style = None
        
        # Control parameters
        self.control_params = {
            'interpolation_method': 'linear',
            'intensity_curve': 'linear',
            'blend_weights': {},
            'current_intensity': 0.5
        }
    
    def add_style_to_library(self, style: StyleVector):
        """Add a style to the library."""
        self.style_library[style.style_name] = style
        print(f"Added style '{style.style_name}' to library")
    
    def create_demo_styles(self) -> Dict[str, StyleVector]:
        """Create demo style vectors for testing."""
        np.random.seed(42)  # For reproducibility
        
        demo_styles = {}
        style_dimension = 128
        
        # Bengali Folk style
        folk_vector = np.random.randn(style_dimension) * 0.5
        folk_vector[:16] *= 2.0  # Emphasize rhythm characteristics
        demo_styles['bengali_folk'] = StyleVector('Bengali Folk', folk_vector)
        
        # Jazz style
        jazz_vector = np.random.randn(style_dimension) * 0.7
        jazz_vector[16:32] *= 1.8  # Emphasize harmonic complexity
        demo_styles['jazz'] = StyleVector('Jazz', jazz_vector)
        
        # Rock style
        rock_vector = np.random.randn(style_dimension) * 0.8
        rock_vector[32:48] *= 2.2  # Emphasize timbral characteristics
        demo_styles['rock'] = StyleVector('Rock', rock_vector)
        
        # Electronic style
        electronic_vector = np.random.randn(style_dimension) * 0.6
        electronic_vector[48:64] *= 1.5  # Emphasize dynamic characteristics
        demo_styles['electronic'] = StyleVector('Electronic', electronic_vector)
        
        # Add to library using the correct keys
        for key, style in demo_styles.items():
            self.style_library[key] = style
        
        return demo_styles
    
    def interactive_interpolation_demo(self, style1_name: str, style2_name: str,
                                     steps: int = 5) -> List[StyleVector]:
        """Demonstrate interactive style interpolation."""
        if style1_name not in self.style_library:
            raise ValueError(f"Style '{style1_name}' not found in library")
        if style2_name not in self.style_library:
            raise ValueError(f"Style '{style2_name}' not found in library")
        
        style1 = self.style_library[style1_name]
        style2 = self.style_library[style2_name]
        
        print(f"\n🎵 Interactive Interpolation: {style1_name} → {style2_name}")
        print(f"=" * 50)
        
        # Generate interpolation sequence
        interpolated = self.interpolator.interpolate_styles(
            style1, style2, steps, 
            method=self.control_params['interpolation_method']
        )
        
        # Display interpolation results
        for i, style in enumerate(interpolated):
            alpha = i / (steps - 1) if steps > 1 else 0
            characteristics = style.get_characteristics()
            
            print(f"Step {i+1}: α={alpha:.2f}")
            print(f"  Style: {style.style_name}")
            print(f"  Rhythm Intensity: {characteristics.get('rhythm_intensity', 0):.3f}")
            print(f"  Harmonic Complexity: {characteristics.get('harmonic_complexity', 0):.3f}")
            print(f"  Timbral Brightness: {characteristics.get('timbral_brightness', 0):.3f}")
            print()
        
        return interpolated
    
    def interactive_blending_demo(self, style_names: List[str],
                                weights: List[float] = None) -> StyleVector:
        """Demonstrate interactive multi-genre blending."""
        if weights is None:
            weights = [1.0 / len(style_names)] * len(style_names)
        
        if len(weights) != len(style_names):
            raise ValueError("Number of weights must match number of styles")
        
        # Get styles from library
        styles = {}
        for name in style_names:
            if name not in self.style_library:
                raise ValueError(f"Style '{name}' not found in library")
            styles[name] = self.style_library[name]
        
        # Create weight dictionary
        weight_dict = dict(zip(style_names, weights))
        
        print(f"\n🎶 Interactive Blending: {' + '.join(style_names)}")
        print(f"=" * 50)
        print(f"Weights: {weight_dict}")
        
        # Create blend
        blended_style = self.blender.blend_styles(styles, weight_dict)
        
        # Display blend characteristics
        characteristics = blended_style.get_characteristics()
        print(f"\nBlended Style: {blended_style.style_name}")
        print(f"Characteristics:")
        for char_name, value in characteristics.items():
            print(f"  {char_name.replace('_', ' ').title()}: {value:.3f}")
        
        return blended_style
    
    def interactive_intensity_demo(self, source_style_name: str,
                                 target_style_name: str,
                                 intensity_levels: List[float] = None) -> List[StyleVector]:
        """Demonstrate interactive intensity control."""
        if intensity_levels is None:
            intensity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        if source_style_name not in self.style_library:
            raise ValueError(f"Style '{source_style_name}' not found in library")
        if target_style_name not in self.style_library:
            raise ValueError(f"Style '{target_style_name}' not found in library")
        
        source_style = self.style_library[source_style_name]
        target_style = self.style_library[target_style_name]
        
        print(f"\n🎛️ Interactive Intensity Control: {source_style_name} → {target_style_name}")
        print(f"=" * 60)
        
        intensity_sequence = self.intensity_controller.create_intensity_sequence(
            source_style, target_style, intensity_levels,
            curve_type=self.control_params['intensity_curve']
        )
        
        # Display intensity results
        for i, (intensity, style) in enumerate(zip(intensity_levels, intensity_sequence)):
            characteristics = style.get_characteristics()
            
            print(f"Intensity Level {i+1}: {intensity:.2f}")
            print(f"  Style: {style.style_name}")
            print(f"  Rhythm Intensity: {characteristics.get('rhythm_intensity', 0):.3f}")
            print(f"  Harmonic Complexity: {characteristics.get('harmonic_complexity', 0):.3f}")
            print(f"  Overall Effect: {'Subtle' if intensity < 0.3 else 'Moderate' if intensity < 0.7 else 'Strong'}")
            print()
        
        return intensity_sequence

def create_demo_interactive_control():
    """Create a comprehensive demonstration of interactive style control."""
    print("Creating Demo Interactive Style Control")
    print("=" * 45)
    
    # Create interactive explorer
    explorer = InteractiveStyleExplorer()
    
    # Create demo styles
    print("1. Creating Demo Style Library...")
    demo_styles = explorer.create_demo_styles()
    
    print(f"   ✅ Created {len(demo_styles)} demo styles:")
    for name in demo_styles.keys():
        print(f"      - {name}")
    
    # Demo 1: Style Interpolation
    print(f"\n2. Demo 1: Style Interpolation")
    interpolated_styles = explorer.interactive_interpolation_demo(
        'bengali_folk', 'jazz', steps=5
    )
    
    # Demo 2: Multi-Genre Blending
    print(f"\n3. Demo 2: Multi-Genre Blending")
    blended_style = explorer.interactive_blending_demo(
        ['bengali_folk', 'jazz', 'rock'], 
        [0.5, 0.3, 0.2]
    )
    
    # Demo 3: Intensity Control
    print(f"\n4. Demo 3: Intensity Control")
    intensity_styles = explorer.interactive_intensity_demo(
        'bengali_folk', 'electronic',
        [0.0, 0.2, 0.5, 0.8, 1.0]
    )
    
    # Demo 4: Advanced Blending with Target Characteristics
    print(f"\n5. Demo 4: Adaptive Blending")
    target_characteristics = {
        'rhythm_intensity': 0.8,
        'harmonic_complexity': 0.6,
        'timbral_brightness': 0.7
    }
    
    adaptive_blend = explorer.blender.adaptive_blend(
        demo_styles, target_characteristics
    )
    
    print(f"Target Characteristics: {target_characteristics}")
    print(f"Adaptive Blend: {adaptive_blend.style_name}")
    adaptive_chars = adaptive_blend.get_characteristics()
    print(f"Achieved Characteristics:")
    for char_name, value in adaptive_chars.items():
        if char_name in target_characteristics:
            target_val = target_characteristics[char_name]
            print(f"  {char_name}: {value:.3f} (target: {target_val:.3f})")
    
    # Save results
    output_dir = "experiments/interactive_control"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'demo_styles': len(demo_styles),
        'interpolation_steps': len(interpolated_styles),
        'blended_style': blended_style.style_name,
        'intensity_levels': len(intensity_styles),
        'adaptive_blend': adaptive_blend.style_name
    }
    
    import json
    results_file = os.path.join(output_dir, "interactive_control_demo.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"\n✅ Demo Interactive Style Control Complete!")
    print(f"🎮 Features: Interpolation, Blending, Intensity, Adaptive Control")
    
    return {
        'explorer': explorer,
        'demo_styles': demo_styles,
        'results': results
    }

if __name__ == "__main__":
    # Run the demonstration
    demo_result = create_demo_interactive_control()
    
    print(f"\n🎨 Interactive Style Control System Ready!")
    print(f"   Capabilities: Multi-genre blending, Real-time control")
    print(f"   Demo styles: {demo_result['results']['demo_styles']} genres available")