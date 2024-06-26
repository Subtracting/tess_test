import pytesseract
import cv2
import numpy as np
from nltk.corpus import wordnet

verbose = 0
language = "eng"
tesseract_models = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def ocr_score(text):
    text = " ".join(text).replace("  ", " ").split(" ")
    total_tokens = len(text)
    known_tokens = sum(1 for token in text if len(wordnet.synsets(token)) == 0)

    validity = known_tokens / total_tokens

    return validity, total_tokens, known_tokens


# DRAW CONTOUR BOXES
def get_contours(gray_input):
    thresh_cont = cv2.threshold(
        gray_input, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    opening = cv2.morphologyEx(
        thresh_cont, cv2.MORPH_OPEN, opening_kernel, iterations=1
    )
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilate = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    dilate = cv2.bitwise_not(dilate)

    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    result = image.copy()

    return contours, result


# WARP MATRIX
def warp_matrix(image, result, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(result, [box], 0, (0, 255, 255), 2)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")

    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        image,
        M,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    if width <= height:
        transposed = cv2.transpose(warped)
        warped = cv2.flip(transposed, 1)

    return warped


# OCR PREPROCESS
def ocr_preprocess(warped):
    gray1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray1, (5, 5), sigmaX=0, sigmaY=0)

    gray = cv2.divide(gray1, blur, scale=255)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh_extended_border = cv2.copyMakeBorder(
        thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    thresh_final = cv2.bitwise_not(thresh_extended_border)
    # thresh = cv2.dilate(thresh, None, iterations=1)

    return thresh_final


for model in tesseract_models:
    config = f"--psm {model}"

    print(config)
    print("-------------------------------------------")

    image = cv2.imread("book.jpg")
    (im_h, im_w) = image.shape[:2]
    cut_h, cut_w = (im_h // 2, im_w // 2)

    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray1, (0, 0), sigmaX=33, sigmaY=33)
    # gray = cv2.divide(gray1, blur, scale=255)

    text = []
    j = 0

    contours, result = get_contours(gray_input=gray1)

    for contour in contours:
        j += 1

        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 1)

        warped = warp_matrix(image=image, result=result, contour=contour)
        thresh_final = ocr_preprocess(warped)

        cv2.imwrite(f"img_{j}.png", thresh_final)

        # OCR IMAGE
        try:
            data = pytesseract.image_to_string(
                thresh_final, lang=language, config=config
            )
            text.append(data)
        except:
            if verbose == 1:
                print(f"OCR with {config} not possible for contour {j}")
            pass

    cv2.imwrite("contours.png", result)
    text_cleaned = []

    for line in text[::-1][1:]:
        line_clean = line.replace("\n", " ")
        if line_clean != "":
            text_cleaned.append(line_clean)

    print(ocr_score(text_cleaned))
