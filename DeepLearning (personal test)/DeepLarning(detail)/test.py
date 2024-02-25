from PIL import Image as PILImage

# Convert the image to a supported format (PNG) before embedding into a PDF
original_image_path = '/mnt/data/Imagine_a_sleek,_modern_developer_portfolio_websit.png'
converted_image_path = '/mnt/data/Developer_Portfolio_Converted.png'

# Open the original image and save it as PNG
image = PILImage.open(original_image_path)
image.save(converted_image_path, format='PNG')

# Try creating the PDF again
pdf = FPDF()

# Add a page
pdf.add_page()

# Set font
pdf.set_font("Arial", size = 12)
pdf.cell(200, 10, txt = "Developer Portfolio", ln = True, align = 'C')

# Add the converted image to the PDF
pdf.image(converted_image_path, x = 10, y = 20, w = 190)

# Save the PDF
pdf_path = "/mnt/data/Developer_Portfolio.pdf"
pdf.output(pdf_path)

pdf_path

