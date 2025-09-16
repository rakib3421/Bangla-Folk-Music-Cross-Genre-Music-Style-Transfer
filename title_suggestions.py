#!/usr/bin/env python3
"""
Simple script to create a professional spectrogram comparison figure
This script creates a clean title for the spectrogram comparison
"""

def create_professional_title_suggestions():
    """
    Provides professional title options for the spectrogram comparison
    """
    
    print("=== PROFESSIONAL TITLE OPTIONS FOR SPECTROGRAM COMPARISON ===")
    print()
    
    title_options = [
        "Spectrogram Comparison: Bengali Folk → Rock Style Transfer",
        "Original vs. Style-Transferred Audio Spectrograms",
        "Cross-Genre Style Transfer: Folk to Rock Transformation",
        "Bengali Folk Music Style Transfer Results",
        "Spectral Analysis: Original and Transformed Audio"
    ]
    
    print("Suggested professional titles:")
    for i, title in enumerate(title_options, 1):
        print(f"{i}. {title}")
    
    print()
    print("=== RECOMMENDED APPROACH ===")
    print("1. Use title: 'Spectrogram Comparison: Bengali Folk → Rock Style Transfer'")
    print("2. Ensure subtitle clearly indicates 'Original' and 'Style Transferred' sections")
    print("3. Use consistent font sizes: Main title (16pt), Subtitles (14pt)")
    print("4. Apply professional color schemes (viridis, plasma, or similar)")
    print()
    
    # Generate HTML for reference
    html_template = """
    <div style="text-align: center; font-family: Arial, sans-serif;">
        <h2 style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">
            Spectrogram Comparison: Bengali Folk → Rock Style Transfer
        </h2>
        <div style="display: flex; flex-direction: column;">
            <div style="margin-bottom: 10px;">
                <h3 style="font-size: 14px; font-weight: bold;">Original: Bengali Folk Music</h3>
            </div>
            <div>
                <h3 style="font-size: 14px; font-weight: bold;">Style Transferred: Folk → Rock</h3>
            </div>
        </div>
    </div>
    """
    
    with open('spectrogram_title_template.html', 'w') as f:
        f.write(html_template)
    
    print("HTML template saved as 'spectrogram_title_template.html' for reference")

if __name__ == "__main__":
    create_professional_title_suggestions()