"""
Combine v4 presentation with Exp 12, 13, 14 8-cut showcase slides into v5.
"""
import os
from pptx import Presentation
from build_exp12_14_slides import create_showcase_presentation

root_dir = os.path.dirname(os.path.abspath(__file__))
v4_path = os.path.join(root_dir, "SD3.5_FewShot_MultiSubject_Presentation_v4.pptx")
showcase_path = os.path.join(root_dir, "SD3.5_Exp12_14_Showcase_Presentation.pptx")

print("Generating SD3.5_Exp12_14_Showcase_Presentation.pptx...")
create_showcase_presentation()
print("Done.")
