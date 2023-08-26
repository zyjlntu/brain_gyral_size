This directory contains files for demonstrating how to measure gyral size using Method 2.

- `whole_brain.jpg`: The picture contains both the brain and the scale bar. This image is from the Mammalian Brain Collections, available at www.brainmuseum.org. It is the property of the University of Wisconsin and the Michigan State Comparative Mammalian Brain Collections, funded by the National Science Foundation and the National Institutes of Health.

- `brain_sample.png`: A screenshot of a portion of the brain to be used for measuring gyral size.

- `scale_bar.png`: A screenshot of the scale bar to be used for measuring the length of the scale bar in pixels.

- `measure_scale_in_pixels.py`: The code for measuring the scale in pixels.

- `measure_gyral_size_in_pixels.py`: The code for measuring gyral size in pixels, with 5 adjustable parameters.

Usage:

To measure the scale in pixels:
python3 measure_scale_in_pixels.py scale_bar.png scale_bar_output.png

To measure gyral size in pixels:
python3 measure_gyral_size_in_pixels.py brain_sample.png brain_sample_output.png

The output values from both programs can be used along with the scale to obtain the gyral size.
