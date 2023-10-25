import pygame
import pygame_gui
import json
import numpy as np
import NN
import operation
pygame.init()

# Set up the Pygame window
WINDOW_SIZE = (pygame.display.Info().current_w,pygame.display.Info().current_h)
screen = pygame.display.set_mode(WINDOW_SIZE, pygame.FULLSCREEN)


image = pygame.transform.scale(
    pygame.image.load("images\penguin.jpg"), (300, 300))
mor_image = pygame.transform.scale(
    pygame.image.load("images\circles.png"), (200, 200))
image_erosion = pygame.transform.scale(
    pygame.image.load("images/background.png"), (200, 200))
num_image = pygame.transform.scale(
    pygame.image.load("images/test7.png"), (600, 200))
image_dilatation = image_erosion.copy()
image_counter = image_erosion.copy()
image_c = image.copy()
image_c_c = image.copy()
img_rect = image.get_rect()
img_rect.topleft = (WINDOW_SIZE[0]//2 - image.get_size()
                    [0]//2, WINDOW_SIZE[1]//2 - image.get_size()[1]//2)

manager = pygame_gui.UIManager(WINDOW_SIZE, "json.json")
surfaces2 = []
brightness_level = 0
scale_level = 1
rotation_factor = 0
write = True

# mask pour tester  la convolution
[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
# mask pour tester  les operateurs morphlogiques
[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
format_images = []


def normalize_images(image):
    image = pygame.transform.scale(image, (28, 28))
    # Convert the image to an array of pixel values
    pixels = pygame.surfarray.array3d(image)
    # Convert the RGB values to grayscale
    gray = (pixels[:, :, 0] + pixels[:, :, 1] + pixels[:, :, 2]) / 3

    # Threshold the grayscale image to create a binary image
    threshold = 40
    binary = (gray >= threshold).astype(int)

    # Convert the binary array to a 1D list and add it to the image_list
    binary_list = binary.ravel().tolist()
    # print(binary_list)
    return binary_list


# def normalize_images1(img):
#     img_split = NN.split_image(img)
#     a = [NN.count_zeros(img_split[0]), NN.count_zeros(
#         img_split[1]), NN.count_zeros(img_split[2]), NN.count_zeros(img_split[3])]
#     return np.array(a)/np.max(a)


# image_list = [pygame.image.load(f"RF/split_digits/{i}..png") for i in range(10)]
image_list = [pygame.image.load(f"split_digits/{i}..png") for i in range(10)]+[pygame.image.load(
    f"split_digits/{i}.png") for i in range(10)]+[pygame.image.load(f"split_digits/{i}...png") for i in range(10)]
for v in image_list:
    format_images.append(normalize_images(v))

X = np.array(format_images)
# print(X.shape)
# print(X)
# y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# # # for i in X :
#     print(i)
# NN entrainement
input_size = 784
hidden_size = 128
output_size = 10
nn = NN.NeuralNetwork(input_size, hidden_size, output_size)
epochs = 500000
nn.train(X, y, epochs)


# Create a dropdown menu
options = ["Choose Op","Convert", "Histogram", "Brightness", "Scale",
           "Rotate", "Convolution", "Morphological op", "chiffre slicing"]
dropdown = pygame_gui.elements.UIDropDownMenu(options_list=options,
                                              starting_option=options[0],
                                              relative_rect=pygame.Rect(
                                                  (WINDOW_SIZE[0]-210, 10), (200, 70)),
                                              manager=manager)

options_morphology = ["choose image", "circles", "geometric formes"]
options_num = ["choose image", "test7",
               "test2", "test3", "test4", "test5", "test6", "test"]
options_other = ["choose image",  "bakri", "wis", 
                 "penguin", "zizo lite", "man with glass"]

dropdown_images = pygame_gui.elements.UIDropDownMenu(options_list=options_other,
                                                     starting_option="choose image",
                                                     relative_rect=pygame.Rect(
                                                         (10, 10), (200, 70)),
                                                     manager=manager)
options_convert = ["choose","BIN","NG","RGB", "HSV", "YCbCr"]
dropdown_convert_from= pygame_gui.elements.UIDropDownMenu(options_list=options_convert,
                                                     starting_option="choose",
                                                     relative_rect=pygame.Rect(
                                                         (400, 550), (100, 50)),
                                                     manager=manager)
dropdown_convert_to= pygame_gui.elements.UIDropDownMenu(options_list=options_convert,
                                                     starting_option="choose",
                                                     relative_rect=pygame.Rect(
                                                         (700, 550), (100, 50)),
                                                     manager=manager)

mask_list_input = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect(
        (img_rect.left, img_rect.bottom+10), (image.get_size()[0], 50)),
    manager=manager
)


submit_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (img_rect.left, img_rect.bottom+70), (image.get_size()[0], 50)),
    text="Submit",
    manager=manager
)
convert_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (img_rect.left+70, img_rect.bottom+70), (80, 30)),
    text="convert",
    manager=manager
)


standart_input1 = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect(
        (img_rect.left+100, img_rect.bottom+10), (100, 30)),
    manager=manager
)
predicted_digits = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect(
        (0, 620), (300, 30)),
    manager=manager
)
plus_button1 = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (img_rect.left+60, img_rect.bottom+10), (40, 30)),
    text="+",
    manager=manager
)

minus_button1 = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (img_rect.left+200, img_rect.bottom+10), (40, 30)),
    text="-",
    manager=manager
)

standart_input2 = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect(
        (img_rect.left+100, img_rect.bottom+150), (100, 30)),
    manager=manager
)

plus_button2 = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (img_rect.left+60, img_rect.bottom+150), (40, 30)),
    text="+",
    manager=manager
)

minus_button2 = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (img_rect.left+200, img_rect.bottom+150), (40, 30)),
    text="-",
    manager=manager
)
back_to_main = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 100), (90, 40)),
    text="<--",
    manager=manager
)
reset = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((1111, 630), (90, 40)),
    text="reset",
    manager=manager
)
dilatation = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((345, WINDOW_SIZE[1]-100), (110, 40)),
    text="dilatation",
    manager=manager
)
erosion = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((545, WINDOW_SIZE[1]-100), (110, 40)),
    text="erosion",
    manager=manager
)
counteur = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((745, WINDOW_SIZE[1]-100), (110, 40)),
    text="counteur",
    manager=manager
)


def turn_on(l):
    list_of_widgets = [mask_list_input, submit_button, standart_input1, plus_button1, minus_button1,
                       standart_input2, plus_button2, minus_button2, dilatation, erosion, counteur, reset, predicted_digits,dropdown_convert_from,dropdown_convert_to,convert_button]
    for i, v in enumerate(list_of_widgets):
        if i in l:
            
            v.show()
        else:
            v.hide()
            


turn_on([])
op = operation.Opertation(screen)

curreent_op = "none"


def reset_fun():
    global write
    global scale_level
    global rotation_factor
    global brightness_level
    global image
    global surfaces2
    global image_dilatation
    global image_counter
    global image_erosion

    surfaces2 = []
    write = True
    scale_level = 1
    rotation_factor = 0
    brightness_level = 0

    image_erosion = pygame.transform.scale(
        pygame.image.load("images/background.png"), (200, 200))
    image_dilatation = image_erosion.copy()
    image_counter = image_erosion.copy()
    image = image_c.copy()


class Stage():
    def __init__(self):
        self.state = "main"

    def main(self):
        if write:
            standart_input1.set_text("     "+str(brightness_level))
            standart_input2.set_text("    {:.1f}".format(
                scale_level) if curreent_op == "Scale" else "    {:.1f}".format(rotation_factor))
        manager.update(pygame.time.Clock().tick(60) / 1000.0)
        img_rect = image.get_rect(
            center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2))
        screen.blit(image, img_rect)
        back_to_main.visible = False
        manager.draw_ui(screen)

    def histogram(self):
        op.draw_histogram(image)
        manager.update(pygame.time.Clock().tick(60) / 1000.0)
        manager.draw_ui(screen)

    def morphological_op(self):
        manager.update(pygame.time.Clock().tick(60) / 1000.0)
        img_rect = mor_image.get_rect(center=(300, 150))
        screen.blit(mor_image, img_rect)
        img_rect = image_dilatation.get_rect(center=(550, 150))
        screen.blit(image_dilatation, img_rect)
        img_rect = image_erosion.get_rect(center=(850, 150))
        screen.blit(image_erosion, img_rect)
        img_rect = image_counter.get_rect(center=(600, 388))
        screen.blit(image_counter, img_rect)
        manager.draw_ui(screen)
    def Convert_op(self):
        manager.update(pygame.time.Clock().tick(60) / 1000.0)
        img_rect = mor_image.get_rect(center=(200, 400))
        screen.blit(mor_image, img_rect)
        img_rect = image_dilatation.get_rect(center=(700, 400))
        screen.blit(image_dilatation, img_rect)
        manager.draw_ui(screen)

    def chiffre_slicing(self):
        predict = ""
        manager.update(pygame.time.Clock().tick(60) / 1000.0)
        img_rect = num_image.get_rect(center=(550, 150))
        screen.blit(num_image, img_rect)
        for i, v in enumerate(surfaces2):
            # pygame.image.save(v, f"split_digits/{i}...png")
            predict = predict+str(nn.predict(normalize_images(v))[0])
            v = pygame.transform.scale(v, (90, 100))
            screen.blit(v, (60+(i*110), 400))
        predicted_digits.set_text(predict)

        manager.draw_ui(screen)


# Game loop
is_running = True
stage = Stage()
while is_running:
    screen.fill((22, 27, 37))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            is_running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:

            write = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if not write:
                    if curreent_op == "Brightness":
                        brightness_level = int(standart_input1.get_text())
                        image = op.brightness(image_c_c, brightness_level)
                        image_c_c = image_c.copy()
                        write = True
                    if curreent_op == "Scale":
                        scale_level = float(standart_input2.get_text())
                        image = op.redimensionner_image(image_c_c, scale_level)
                        image_c_c = image_c.copy()
                        write = True
                    if curreent_op == "Rotate":
                        rotation_factor = float(standart_input2.get_text())
                        image = op.rotate_image(image_c_c, rotation_factor)
                        image_c_c = image_c.copy()
                        write = True

        elif event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:

                if event.ui_element == plus_button1:
                    write = True
                    brightness_level += 5
                    image = op.brightness(image, 5)
                if event.ui_element == minus_button1:
                    write = True
                    brightness_level -= 5
                    image = op.brightness(image, -5)
                # brightness events
                if event.ui_element == plus_button2:
                    write = True
                    if curreent_op == "Scale":
                        scale_level += .1
                        image = op.redimensionner_image(image_c, scale_level)
                    else:
                        rotation_factor += 10
                        if rotation_factor == 360 or rotation_factor == -360:
                            rotation_factor = 0
                        image = op.rotate_image(image_c, rotation_factor)

                if event.ui_element == minus_button2:
                    write = True
                    if curreent_op == "Scale":
                        scale_level -= .1
                        image = op.redimensionner_image(image_c, scale_level)
                    else:
                        rotation_factor -= 10
                        image = op.rotate_image(image_c, rotation_factor)

                if event.ui_element == submit_button:
                    if curreent_op == "Convolution":
                        mask_list = json.loads(mask_list_input.text)
                        s = pygame.Surface(image_c.get_size())
                        image = op.convolution(image, s, mask_list)
                    else:
                        surfaces = []
                        for i in op.split_vertical(num_image):
                            surfaces.append(pygame.surfarray.make_surface(i))

                        list_finale = []
                        for i in surfaces:
                            list_finale.append(op.split_horizontal(i))
                        for i in list_finale:
                            for j in i:
                                surfaces2.append(pygame.transform.rotate(pygame.transform.flip(
                                    pygame.surfarray.make_surface(j), False, True), -90))

                if event.ui_element == back_to_main:
                    stage.state = "main"
                    dropdown.visible = True
                if event.ui_element == reset:
                    reset_fun()
                if event.ui_element == dilatation:
                    mask_list = json.loads(mask_list_input.text)
                    image_dilatation = op.dilatation(mor_image, mask_list)
                if event.ui_element == erosion:
                    mask_list = json.loads(mask_list_input.text)
                    image_erosion = op.erosion(mor_image, mask_list)

                if event.ui_element == counteur:
                    mask_list = json.loads(mask_list_input.text)
                    image_counter = op.counteur(mor_image, mask_list)

            elif event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == dropdown_images:
                    if curreent_op == "chiffre slicing":
                        if event.text != "choose image":
                            reset_fun()
                            num_image = pygame.transform.scale(
                                    pygame.image.load(f"images\{event.text}.jpg"), (600, 200))  

                    elif curreent_op == "Morphological op":
                        if event.text != "choose image":
                            reset_fun()
                            
                            mor_image = pygame.transform.scale(
                                pygame.image.load(f"images\{event.text}.png"), (200, 200))

                    else:
                        if event.text != "choose image":
                            reset_fun()
                            image = pygame.transform.scale(pygame.image.load(
                                f"images\{event.text}.jpg"), (300, 300))
                            image_c = image.copy()
                            image_c_c = image.copy()
                else:
                    if event.text == "Brightness":
                        dropdown_images.selected_option = "choose image"
                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_other)

                        stage.state = "main"
                        reset_fun()
                        curreent_op = "Brightness"
                        turn_on([2, 3, 4, 11])
                    if event.text=="Convert":
                        dropdown_images.selected_option = "choose image"
                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_other)
                        stage.state = "Convert"
                        reset_fun()
                        curreent_op = "Convert"
                        
                        turn_on([13,14,15])
                    if event.text == "Convolution":
                        dropdown_images.selected_option = "choose image"
                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_other)

                        stage.state = "main"
                        reset_fun()
                        curreent_op = "Convolution"
                        turn_on([0, 1, 11])
                    if event.text == "Scale":
                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_other)
                        dropdown_images.selected_option = "choose image"
                        stage.state = "main"
                        reset_fun()
                        curreent_op = "Scale"
                        turn_on([5, 6, 7, 11])
                    if event.text == "Rotate":
                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_other)
                        dropdown_images.selected_option = "choose image"
                        stage.state = "main"
                        reset_fun()
                        curreent_op = "Rotate"
                        turn_on([5, 6, 7, 11])
                    if event.text == "Histogram":
                        stage.state = "histogram"
                        back_to_main.visible = True
                        dropdown.visible = False
                        turn_on([11])
                    if event.text == "Morphological op":
                        curreent_op = "Morphological op"
                        dropdown_images.selected_option = "choose image"
                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_morphology)
                        stage.state = "morphological op"

                        turn_on([0, 8, 9, 10])
                    if event.text == "chiffre slicing":

                        dropdown_images.remove_options(
                            dropdown_images.options_list)
                        dropdown_images.add_options(options_num)
                        dropdown_images.selected_option = "choose image"

                        stage.state = "chiffre slicing"
                        curreent_op = "chiffre slicing"
                        turn_on([1, 12])

        manager.process_events(event)

    if stage.state == "main":
        stage.main()
    elif stage.state == "histogram":
        stage.histogram()
    elif stage.state == "morphological op":
        stage.morphological_op()
    elif stage.state == "Convert":
        stage.Convert_op()
    elif stage.state == "chiffre slicing":
        stage.chiffre_slicing()
    pygame.display.update()

pygame.quit()
