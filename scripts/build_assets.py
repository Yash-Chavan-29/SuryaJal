import os
import sys

sys.path.insert(0, ".")
import vision_scout

os.makedirs("assets/samples", exist_ok=True)
vision_scout.generate_sample_field_image("cotton").save("assets/samples/cotton.jpg")
vision_scout.generate_sample_field_image("sugarcane").save("assets/samples/sugarcane.jpg")
vision_scout.generate_sample_field_image("chickpea").save("assets/samples/chickpea.jpg")
print("Saved 3 sample farm photos to assets/samples/")
