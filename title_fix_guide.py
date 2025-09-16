"""
Professional title recommendations for spectrogram comparison image
"""

# Recommended professional titles (without Unicode arrows):
PROFESSIONAL_TITLES = [
    "Spectrogram Comparison: Bengali Folk to Rock Style Transfer",
    "Original vs. Style-Transferred Audio Spectrograms", 
    "Cross-Genre Style Transfer: Folk to Rock Transformation",
    "Bengali Folk Music Style Transfer Results",
    "Spectral Analysis: Original and Transformed Audio"
]

# Best title for IEEE paper
RECOMMENDED_TITLE = "Spectrogram Comparison: Bengali Folk to Rock Style Transfer"

# Layout specifications
TITLE_SPECS = {
    "main_title_size": 16,
    "subtitle_size": 14, 
    "main_title_weight": "bold",
    "subtitle_weight": "bold",
    "font_family": "Arial, sans-serif"
}

print("=== FIXING SPECTROGRAM IMAGE TITLE ===")
print()
print("Current corrupted title: 'SpecDiagramComparisowStyle Transfer Results'")
print()
print("RECOMMENDED PROFESSIONAL TITLE:")
print(f"'{RECOMMENDED_TITLE}'")
print()
print("SUBTITLE STRUCTURE:")
print("- Top panel: 'Original: Bengali Folk Music'")
print("- Bottom panel: 'Style Transferred: Folk to Rock'")
print()
print("SPECIFICATIONS:")
print(f"- Main title: {TITLE_SPECS['main_title_size']}pt, {TITLE_SPECS['main_title_weight']}")
print(f"- Subtitles: {TITLE_SPECS['subtitle_size']}pt, {TITLE_SPECS['subtitle_weight']}")
print(f"- Font: {TITLE_SPECS['font_family']}")
print()
print("This title is:")
print("✓ Professional and clear")
print("✓ Suitable for IEEE conference paper")
print("✓ Describes exactly what the figure shows")
print("✓ Avoids special characters that might cause issues")