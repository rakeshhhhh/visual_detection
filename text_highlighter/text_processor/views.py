from django.shortcuts import render
from django.core.files.storage import default_storage, FileSystemStorage
from PIL import Image, ImageDraw, ImageFont
import boto3
from .constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, GROQ_API_KEY, GOOGLE_API_KEY
from groq import Groq
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from django.core.files.base import ContentFile
from io import BytesIO
from skimage.metrics import structural_similarity as compare_ssim
import google.generativeai as genai
from IPython.display import Markdown
from django.http import HttpResponse
from django.template.loader import render_to_string
from xhtml2pdf import pisa
from django.urls import reverse


# Initialize AWS Textract client
textract_client = boto3.client('textract', region_name='ap-south-1',
                               aws_access_key_id=AWS_ACCESS_KEY,
                               aws_secret_access_key=AWS_SECRET_KEY)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")


def index(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            image_path = fs.save(f'images/{image_file.name}', image_file)
            image_url = fs.url(image_path)

            # Extract text from the image using AWS Textract
            with fs.open(image_path, 'rb') as img_file:
                img_data = bytearray(img_file.read())

            response = textract_client.detect_document_text(
                Document={'Bytes': img_data}
            )

            extracted_text = " ".join(
                item["Text"] for item in response["Blocks"] if item["BlockType"] == "LINE"
            )

            # Function to correct spelling using the Llama model through Groq API
            corrected_sentence = correct_spelling(extracted_text)

            # Highlight mistakes in the image
            highlighted_image_path = highlight_mistakes_in_image(image_path, extracted_text, corrected_sentence, response)
            highlighted_image_url = fs.url(highlighted_image_path)

            context = {
                'image_url': image_url,
                'extracted_text': extracted_text,
                'corrected_text': corrected_sentence,
                'highlighted_image_url': highlighted_image_url,
            }

            if 'c_image' in request.FILES:
                c_image_file = request.FILES['c_image']
                c_image_path = fs.save(f'images/{c_image_file.name}', c_image_file)
                c_image_url = fs.url(c_image_path)

                # Compare the images
                diff_image_path1, diff_image_path2, diff_image_combined_path = compare_images(image_path, c_image_path)

                # Call the spot_differences function to analyze visual differences using the Gemini model
                differences_text = spot_differences(image_path, c_image_path)

                context.update({
                    'c_image_url': c_image_url,
                    'diff_image_url1': fs.url(diff_image_path1),
                    'diff_image_url2': fs.url(diff_image_path2),
                    'diff_combined_url': fs.url(diff_image_combined_path),
                    'differences_text': differences_text,
                })

            return render(request, 'text_processor/index.html', context)

        elif 'text' in request.POST:
            user_text = request.POST['text']
            corrected_text = correct_spelling(user_text)
            incorrect_words = find_incorrect_words(user_text, corrected_text)

            return render(request, 'text_processor/index.html', {
                'user_text': user_text,
                'corrected_text': corrected_text,
                'incorrect_words': incorrect_words,
            })

    return render(request, 'text_processor/index.html')


def spot_differences(image1_path, image2_path):
    # Open the images using a context manager
    with default_storage.open(image1_path, 'rb') as img_file1, \
         default_storage.open(image2_path, 'rb') as img_file2:
        img1_cv2 = cv2.imdecode(np.frombuffer(img_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2_cv2 = cv2.imdecode(np.frombuffer(img_file2.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert OpenCV images to PIL format
        img1 = cv2.cvtColor(img1_cv2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PIL
        img1 = Image.fromarray(img1)
        img2 = cv2.cvtColor(img2_cv2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)

    # Prepare the prompt for Gemini model
    prompt = "Compare the two images and highlight the visual differences, such as background color, wrong text font, overlapping texts, misplaced elements, etc."

    # Call the Gemini model
    response = gemini_model.generate_content([prompt, img1, img2])

    # Extract the textual description of the differences from the response
    differences_text = response.text.strip()  # Remove any leading/trailing whitespace
    differences_lines = differences_text.split("\n")  # Split into lines

    # Create HTML formatted bullet points
    # Remove asterisks from each line and add numbering
    html_output = "<ol>\n"  # Ordered list in HTML (for numbered list)
    for idx, line in enumerate(differences_lines, 1):
        clean_line = line.replace("*", "").strip()  # Remove asterisks and extra whitespace
        html_output += f"<li>{clean_line}</li>\n"  # Each line becomes a numbered list item
    html_output += "</ol>"

    return html_output


def compare_images(image1_path, image2_path):
    # Load the two images
    with default_storage.open(image1_path, 'rb') as img_file1, default_storage.open(image2_path, 'rb') as img_file2:
        img1 = cv2.imdecode(np.frombuffer(img_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img_file2.read(), np.uint8), cv2.IMREAD_COLOR)

    # Compute the difference between the two images
    difference = cv2.subtract(img1, img2)

    # Convert the difference to grayscale
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary mask of the differences
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Color the differences in red
    difference[mask != 255] = [0, 0, 255]

    # Add the red mask to the original images
    img1[mask != 255] = [0, 0, 255]
    img2[mask != 255] = [0, 0, 255]

    # Save the images with differences highlighted
    diff_image_path1 = os.path.join('images', 'diff_image1_' + os.path.basename(image1_path))
    diff_image_path2 = os.path.join('images', 'diff_image2_' + os.path.basename(image2_path))
    diff_image_combined_path = os.path.join('images', 'diff_combined_' + os.path.basename(image1_path))

    # Convert images from BGR to RGB for saving as PIL images
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    difference_rgb = cv2.cvtColor(difference, cv2.COLOR_BGR2RGB)

    # Save the images
    with default_storage.open(diff_image_path1, 'wb') as img_file1, \
         default_storage.open(diff_image_path2, 'wb') as img_file2, \
         default_storage.open(diff_image_combined_path, 'wb') as img_file_combined:

        img1_pil = Image.fromarray(img1_rgb)
        img2_pil = Image.fromarray(img2_rgb)
        combined_pil = Image.fromarray(difference_rgb)

        img1_pil.save(img_file1, format='PNG')
        img2_pil.save(img_file2, format='PNG')
        combined_pil.save(img_file_combined, format='PNG')

    return diff_image_path1, diff_image_path2, diff_image_combined_path


def correct_spelling(user_sentence):
    completion = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "\n"},
            {"role": "user", "content": f"Correct the spelling mistake: {user_sentence}"}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    corrected_text = ""
    for chunk in completion:
        corrected_text += chunk.choices[0].delta.content or ""

    return corrected_text


def find_incorrect_words(original_text, corrected_text):
    original_words = set(original_text.split())
    corrected_words = set(corrected_text.split())
    return list(original_words.difference(corrected_words))


def highlight_mistakes_in_image(image_path, extracted_text, corrected_text, response):
    # Open the image again (ensuring it is not closed prematurely)
    with default_storage.open(image_path, 'rb') as img_file:
        image = Image.open(img_file)
        image = image.convert("RGBA")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    extracted_words = extracted_text.split()
    corrected_words = corrected_text.split()
    incorrect_words = set(extracted_words).difference(set(corrected_words))

    for item in response["Blocks"]:
        if item["BlockType"] == "WORD" and item["Text"] in incorrect_words:
            bbox = item["Geometry"]["BoundingBox"]
            width, height = image.size
            x = int(bbox["Left"] * width)
            y = int(bbox["Top"] * height)
            w = int(bbox["Width"] * width)
            h = int(bbox["Height"] * height)

            draw.rectangle([(x, y), (x + w, y + h)], outline="yellow", width=3, fill=(255, 255, 0, 100))
            draw.text((x, y - 10), item["Text"], fill="black", font=font)

    highlighted_image_path = os.path.join('images', 'highlighted_' + os.path.basename(image_path))

    # Save the highlighted image correctly
    with default_storage.open(highlighted_image_path, 'wb') as img_file:
        image.save(img_file, format='PNG')

    return highlighted_image_path


def generate_pdf(request):
    if request.method == 'POST':
        # Retrieve the image URLs from the POST data and convert them to absolute URLs
        image_url = request.build_absolute_uri(request.POST.get('image_url', ''))
        highlighted_image_url = request.build_absolute_uri(request.POST.get('highlighted_image_url', ''))
        c_image_url = request.build_absolute_uri(request.POST.get('c_image_url', ''))
        diff_image_url1 = request.build_absolute_uri(request.POST.get('diff_image_url1', ''))
        diff_image_url2 = request.build_absolute_uri(request.POST.get('diff_image_url2', ''))
        diff_combined_url = request.build_absolute_uri(request.POST.get('diff_combined_url', ''))

        # Context for the PDF
        context = {
            'user_text': request.POST.get('user_text', ''),
            'incorrect_words': request.POST.getlist('incorrect_words'),
            'image_url': image_url,
            'extracted_text': request.POST.get('extracted_text', ''),
            'corrected_text': request.POST.get('corrected_text', ''),
            'highlighted_image_url': highlighted_image_url,
            'c_image_url': c_image_url,
            'diff_image_url1': diff_image_url1,
            'diff_image_url2': diff_image_url2,
            'diff_combined_url': diff_combined_url,
            'differences_text': request.POST.get('differences_text', ''),
        }

        # Render the template to HTML
        html_string = render_to_string('text_processor/pdf_template.html', context)

        # Create a BytesIO buffer to receive the PDF data
        result = BytesIO()

        # Convert HTML to PDF
        pdf = pisa.CreatePDF(BytesIO(html_string.encode("UTF-8")), dest=result)

        # Check for errors
        if pdf.err:
            return HttpResponse('We had some errors <pre>' + html_string + '</pre>')

        # Return the PDF as an HTTP response
        result.seek(0)
        return HttpResponse(result, content_type='application/pdf')

    return HttpResponse('Invalid request method', status=405)