#! python3
"""
todo:
    packaging
        + tidy up the code
        + add necessary documentation
    graphic
        + add fret index header
        + add fretboard markers
    tuning
        + add add additional tuning to choose from, drop d, drop b ect, jose gonzales, mastodon
        + allow the user to manually specify the guitar tuning
"""
import copy
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Any, Collection, Iterator, Union

from PIL import Image, ImageFont, ImageDraw

logger = logging.getLogger(__name__)

LOG_FILE = 'logs.txt'
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.DEBUG,
    filename=LOG_FILE,
)

Rgb = Tuple[int, int, int]
Img = Image.Image

ROOT = Path.cwd().parent
OUTPUT = ROOT / 'output'

WIDTH: int
HEIGHT: int
WIDTH, HEIGHT = (1500, 600)

white: Rgb = (255, 255, 255)
black: Rgb = (0, 0, 0)

scale_tones: Dict[str, List[int]] = {
    'chromatic': list(range(1, 12)),
    'major': [2, 2, 1, 2, 2, 2, 1],
    'minor': [2, 1, 2, 2, 1, 2, 2],
    "lonian": [2, 2, 1, 2, 2, 2, 1],
    "Dorian": [2, 1, 2, 2, 2, 1, 2],
    "Phrygian": [1, 2, 2, 2, 1, 2, 2],
    "Lydian": [2, 2, 2, 1, 2, 2, 1],
    "Mixolydian": [2, 2, 1, 2, 2, 1, 2],
    "Aeolian": [2, 1, 2, 2, 1, 2, 2],
    "Locrian": [1, 2, 2, 1, 2, 2, 2],
}


def clear_log() -> None:
    """clear contents of the log file"""
    with open(LOG_FILE, 'w'):
        pass


def decorator_log_output(msg_prefix: str):
    """a decorator that logs the return value prefixed with a custom message"""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.debug(msg_prefix + str(result))
            return result

        return inner_wrapper

    return outer_wrapper


@dataclass
class Coordinates:
    x: float
    y: float

    def get_xy(self) -> Tuple[float, float]:
        # noinspection PyRedundantParentheses
        return (self.x, self.y)


@dataclass
class Dimensions:
    width: float
    height: float

    @property
    def width_by_height(self) -> Tuple[float, float]:
        return (self.width, self.height)


@dataclass
class GeometricRectangle:
    """the geometric properties of a rectangle that will be represented on a 2d grid"""
    top_left: Coordinates
    dimensions: Dimensions

    def get_bottom_right(self) -> Coordinates:
        return Coordinates(
            x=self.top_left.x + self.dimensions.width,
            y=self.top_left.y + self.dimensions.height
        )

    def get_bottom_left(self) -> Coordinates:
        return Coordinates(
            x=self.top_left.x,
            y=self.top_left.y + self.dimensions.height,
        )

    def get_top_right(self) -> Coordinates:
        return Coordinates(
            x=self.top_left.x + self.dimensions.width,
            y=self.top_left.y,
        )

    def from_coordinates(self, top_left: Coordinates):
        """constructs another rectangle from upper left coordinates with the same dimensions as the instance"""
        return GeometricRectangle(top_left=top_left, dimensions=self.dimensions)


# select number of strings and return the dimensions of the fretboard, outputs:
#   num strings
#   image size
#   fretsize

def get_tuning_standard() -> List[str]:
    """return opening tuning as a list high string to low string"""
    return ['E', 'B', 'G', 'D', 'A', 'E', 'B', 'F#']


def get_chromatic_notes(key: str) -> List[str]:
    """return a chromatic scale in a certain key"""
    chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', ]
    try:
        chromatic_notes_ordered = reindex_lst(chromatic_notes, key)
    except ValueError:
        raise ValueError(f'enter a valid key: {" ".join(chromatic_notes)}')
    return chromatic_notes_ordered


def input_num_gstrings() -> int:
    """ask the user for the number of strings, input will only accept a number between 6 and 8"""
    while True:
        user_strings = input("How many strings? ")
        try:
            num_strings = int(user_strings.strip())
        except ValueError:
            print('Invalid answer, please try again...\n')
            continue
        except Exception:
            logger.debug(f'User input raised an exception: {user_strings}')
            raise Exception('Something went wrong!')
        if num_strings not in [6, 7, 8]:
            print('Please choose a number from 6 to 8...\n')
            continue
        else:
            break

    assert num_strings is not None
    return num_strings


class ImagePanes:

    def __init__(self, width: int, height: int, num_strings: int):
        self.width = width
        self.height = height
        self.num_strings = num_strings

        self.fretboard_proportions = Dimensions(0.85, 0.75)
        self.pane_proportions = {
            'title': Dimensions(1, 0.15),
            'index_gstrings': Dimensions(1 - self.fretboard_proportions.width, self.fretboard_proportions.height),
            'fretboard': self.fretboard_proportions,
            'index_frets': Dimensions(self.fretboard_proportions.width, 0.1),
        }
        err_msg: Callable = lambda i: f"{i} pane proportions do not sum to 1"
        assert (self.fretboard_proportions.width +
                self.pane_proportions['index_gstrings'].width) == 1.0, err_msg('width')
        assert (self.pane_proportions['title'].height + self.fretboard_proportions.height +
                self.pane_proportions['index_frets'].height) == 1.0, err_msg('height')

    def get_img_dimensions(self) -> Dimensions:
        """get the dimensions of the whole image that will be produced factoring in the number of additional guitar
        strings"""
        additional_height_per_extra_gstring = self.height * (1 / 7)
        num_extra_gstrings = self.num_strings - 6
        height_adjusted = self.height + (num_extra_gstrings * additional_height_per_extra_gstring)
        return Dimensions(width=self.width, height=height_adjusted)

    def get_pane_rectangle(self, coordinates: Coordinates, proportions: Dimensions) -> GeometricRectangle:
        """
        get a pane of the image as a geometric rectangle from the top left coordinates and the width and height
        as a proportion of the total image
        Args:
            coordinates: top left corner coordinates of the pane
            proportions: width and height of the pane as a proportion of the total image width

        Returns:
            the new pane dimensions as a Geometric Rectangle object
        """
        pane_width = self.get_img_dimensions().width * proportions.width
        pane_height = self.get_img_dimensions().height * proportions.height
        pane_dimensions = Dimensions(pane_width, pane_height)
        return GeometricRectangle(top_left=coordinates, dimensions=pane_dimensions)

    @decorator_log_output('title_rectangle: '.ljust(30))
    def get_title_pane_rectangle(self) -> GeometricRectangle:
        """"""
        return self.get_pane_rectangle(
            coordinates=Coordinates(0, 0),
            proportions=self.pane_proportions['title']
        )

    @decorator_log_output('gstring_index_pane_rectangle: '.ljust(30))
    def get_gstring_index_pane_rectangle(self) -> GeometricRectangle:
        """"""
        return self.get_pane_rectangle(
            coordinates=self.get_title_pane_rectangle().get_bottom_left(),
            proportions=self.pane_proportions['index_gstrings']
        )

    @decorator_log_output('fretboard pane rectangle: '.ljust(30))
    def get_fretboard_pane_rectangle(self) -> GeometricRectangle:
        """"""
        return self.get_pane_rectangle(
            coordinates=self.get_gstring_index_pane_rectangle().get_top_right(),
            proportions=self.pane_proportions['fretboard']
        )

    @decorator_log_output('fret index pane rectangle: '.ljust(30))
    def get_fret_index_pane_rectangle(self) -> GeometricRectangle:
        """"""
        return self.get_pane_rectangle(
            coordinates=self.get_fretboard_pane_rectangle().get_bottom_left(),
            proportions=self.pane_proportions['index_frets']
        )


def get_fret_dimensions(fretboard_dimensions: Dimensions, num_strings: int, num_notes: int) -> Dimensions:
    """calculate the dimensions of each fret on the graphic considering the additional number index running
     horizontally along the bottom of the graphic"""
    fret_height = fretboard_dimensions.height / num_strings
    fret_width = fretboard_dimensions.width / num_notes
    return Dimensions(fret_width, fret_height)


def infinite_chromatic_note_generator(chromatic_notes: List[str]) -> Iterator:
    """yields successive chromatic notes"""
    while True:
        iter_notes = iter(chromatic_notes)
        for i in iter_notes:
            yield i


def get_next_chromatic_note(note: str, chromatic_notes: List[str]) -> str:
    """get the next note in the chromatic scale"""
    chromatic_generator = infinite_chromatic_note_generator(chromatic_notes)
    for note_generated in chromatic_generator:
        if note == note_generated:
            return next(chromatic_generator)


def get_notes_of_next_fret(notes: List[str], chromatic_notes: List[str]) -> List[str]:
    """takes a list of notes and returns a list where each note is increased by one semi-tone"""
    return [get_next_chromatic_note(note, chromatic_notes) for note in notes]


# get user to select mode and return the corresponding intervals


def show_scales() -> None:
    """print available scales to the terminal"""
    print("Available scales:")
    for scale in scale_tones.keys():
        print(f'\t{scale}')


def input_select_scale() -> str:
    """return the scale selected by the user as a string"""
    while True:
        user_scale = input('Type the name of your chosen scale: ').strip()
        try:
            scale_tones[user_scale]
        except KeyError:
            print('Invalid user input please enter one of the available scales below.\n')
            show_scales()
            continue
        except Exception:
            err_msg = f'something has gone wrong with the scales user input: {user_scale}'
            logger.critical(err_msg)
            raise Exception(err_msg)
        else:
            break
    return user_scale


def cumulative_lst(lst) -> List[float]:
    """return the cumulative sums"""
    cums = []
    i = 0
    for x in lst:
        i = x + i
        cums.append(i)
    return cums


# pick the key of the scale and return the notes of the scale corresponding to that key, outputs:
#   key
#   chromatic scales starting from key


def show_keys(notes: List[str]) -> None:
    print("Available keys: " + ' '.join(notes))


def input_select_key(chromatic_notes: List[str]) -> str:
    """returns the key specified by the user as a string"""
    while True:
        user_key = input('Type the key: ').strip()
        if user_key in chromatic_notes:
            break
        else:
            print('Invalid user input, please select one of the available keys...')
            show_keys(chromatic_notes)
            continue
    return user_key


def reindex_lst(lst, index) -> List:
    """make index the first item of the list whilst preserving the order of elements"""
    pos = lst.index(index)
    start, tail = lst[:pos], lst[pos:]
    tail.extend(start)
    assert len(lst) == len(tail)
    return tail


def lst_multiindex(lst, index: List[int]):
    """return items in lst corresponding to the indexes in index"""
    return [lst[i] for i in index]


def get_scale_notes(scale: str, scale_chromatic: List[str]) -> List[str]:
    """
    return a list of notes corresponding to the selected scale
    Args:
        scale: the mode selected by the user eg major, minor
        scale_chromatic: must be passed in the user selected key # this is ambiguous a test for this or key needs to
         also be passed in as an arg

    Returns:
        a list of notes corresponding to a specific key and mode
    """
    scale_index = tones_to_scale_index(scale_tones[scale])
    try:
        # noinspection PyTypeChecker
        user_scale_notes = lst_multiindex(scale_chromatic, scale_index)
    except TypeError:
        raise TypeError(f'index contains non-int values: {scale_index}')
    return user_scale_notes


def tones_to_scale_index(tones: List[int]) -> List[float]:
    """takes in a list of numerical tones and semi tones returning a list of numerical
    indexes which can be used to index the corresponding scale notes from a chromatic scale"""
    tones.insert(0, 0)
    tones = tones[:-1]
    return cumulative_lst(tones)


# creating the graphic


def _draw_rectangle(img: Image, rectangle: GeometricRectangle, fill: Any, text: Optional[str] = None,
                    outline: Any = 'black', outline_width: float = 3, font_size: float = 50, *args, **kwargs) -> Img:
    """draw a rectangle on an image and return the image, you can add text that will be aligned in the center of
    the rectangle"""
    draw = ImageDraw.Draw(img)
    pt1 = rectangle.top_left
    pt2 = rectangle.get_bottom_right()
    draw.rectangle(xy=(pt1.get_xy(), pt2.get_xy()), fill=fill, outline=outline, width=outline_width, *args, **kwargs)

    if text:
        font = ImageFont.truetype("arial", size=font_size)
        text_width, text_height = draw.textsize(text, font=font)
        x_text = ((rectangle.dimensions.width - text_width) / 2) + pt1.x
        y_text = ((rectangle.dimensions.height - text_height) / 2) + pt1.y
        draw.text((x_text, y_text), text=text, font=font, fill='black')

    return img


def add_tonic_note(img: Image, rectangle: GeometricRectangle, note: str, *args, **kwargs) -> Img:
    """add note graphic to the fret board with text"""
    return _draw_rectangle(img=img, rectangle=rectangle, fill='blue', text=note, *args, **kwargs)


def add_active_note(img: Image, rectangle: GeometricRectangle, note: str, *args, **kwargs) -> Img:
    """add note graphic to the fret board with text"""
    return _draw_rectangle(img=img, rectangle=rectangle, fill='white', text=note, *args, **kwargs)


def add_inactive_note(img: Image, rectangle: GeometricRectangle) -> Img:
    """add note graphic without text"""
    return _draw_rectangle(img=img, rectangle=rectangle, fill='black')


def add_text(img: Image, rectangle: GeometricRectangle, text: str, *args, **kwargs) -> Img:
    """add note graphic to the fret board with text"""
    return _draw_rectangle(img=img, rectangle=rectangle, fill='white', text=text, outline='white', *args, **kwargs)


def get_blank_img(width: float = 1200, height: float = 500) -> Img:
    """create a new blank png file"""
    width, height = round(width), round(height)
    img = Image.new('RGB', size=(width, height), color=(255, 255, 255))
    return img


def get_gtuning(num_gstrings: int, gtuning_8strings: List[str]) -> List[
    str]:  # it might make more sense for the user to pick the the number of strings then the tuning?
    """get the open notes that correspond to the number of strings for the selected tuning"""
    max_num_gstrings = len(gtuning_8strings)
    slice_index = max_num_gstrings - (max_num_gstrings - num_gstrings)
    gtuning_active = gtuning_8strings[:slice_index]
    logger.debug(f'active tuning: {gtuning_active}')
    return gtuning_active


def get_fret_coordinates_for_single_gstring(
    first_fret: GeometricRectangle) -> Dict[str, List[GeometricRectangle]]:
    """"""
    frets = [first_fret]
    x_next_fret = first_fret.get_bottom_right().x

    for i in range(11):
        next_fret_top_left_coordinates = Coordinates(x=x_next_fret, y=first_fret.top_left.y)

        frets.append(
            GeometricRectangle(top_left=next_fret_top_left_coordinates, dimensions=first_fret.dimensions)
        )
        x_next_fret += first_fret.dimensions.width

    return {
        'fret_rectangles': frets,
    }


def get_gstring_struct(open_note: str, rectangle_first_fret: GeometricRectangle, chromatic_notes: List[str]) -> Dict:
    """combine open string details with rectangle coordinates"""
    details = {
        'name': open_note,
        'chromatic': reindex_lst(chromatic_notes, open_note),
    }
    coordinates = get_fret_coordinates_for_single_gstring(first_fret=rectangle_first_fret)
    assert len(coordinates['fret_rectangles']) == len(details['chromatic'])
    return {**details, **coordinates}


def get_fretboard_struct(gtuning: List[str], first_fret: GeometricRectangle) -> List[Dict]:
    """
    create a data structure containing the data required to plot each fret for each string
    Args:
        gtuning: the open notes that correspond to each string
        first_fret: the rectangle geometries for the first fret of the first string

    Returns:
        a list of where each item corresponds to a guitar string
    """
    fretboard_struct = []
    for note in gtuning:
        chromatic_notes_of_string = get_chromatic_notes(note)
        gstring_struct = get_gstring_struct(open_note=note,
                                            rectangle_first_fret=first_fret,
                                            chromatic_notes=chromatic_notes_of_string)
        fretboard_struct.append(gstring_struct)

        first_fret_coordinates_next_string = Coordinates(x=first_fret.top_left.x,
                                                         y=first_fret.top_left.y + first_fret.dimensions.height)
        first_fret = first_fret.from_coordinates(first_fret_coordinates_next_string)
    return fretboard_struct


def populate_fretboard(img: Image, fretboard_struct: List[Dict], user_scale_notes: List[str]) -> Img:
    """
    adds all the fret graphics to the fret board and returns the image
    Args:
        img: the image you are adding the graphics to
        fretboard_struct: all the data ie coordinates required to plot the fret graphics for each string
        user_scale_notes: the notes of the scale that correspond to the key and mode selected by the user

    Returns:
        the edited image object
    """
    tonic_note = user_scale_notes[0]
    for gstring in fretboard_struct:
        for note, fret in zip(gstring['chromatic'], gstring['fret_rectangles']):
            if note == tonic_note:
                img = add_tonic_note(img, rectangle=fret, note=note)
            elif note in user_scale_notes:
                img = add_active_note(img, rectangle=fret, note=note)
            else:
                img = add_inactive_note(img, rectangle=fret)
    return img


def save_img(img: Image, file: str) -> None:
    """save img as a png to the name file"""
    img.save(file, 'PNG')


class InvalidDirectionError(ValueError):
    pass


def get_textbox_rectangle_from_pane(pane_rectangle: GeometricRectangle, texts: Collection[str],
                                    direction: str) -> GeometricRectangle:
    """

    Args:
        pane_rectangle:
        texts:
        direction:

    Returns:

    """
    num_boxes: int = len(texts)
    dimensions = copy.deepcopy(pane_rectangle.dimensions)

    if direction == 'right':
        dimensions.width /= num_boxes
    elif direction == 'down':
        dimensions.height /= num_boxes
    else:
        raise InvalidDirectionError(f'direction must be "right" or "down": {direction}')

    return GeometricRectangle(top_left=pane_rectangle.top_left,
                              dimensions=dimensions)


def add_sequence_of_contiguous_textboxes(img: Image, pane_rectangle: GeometricRectangle, texts: Collection[str],
                                         direction: str) -> Img:
    """
    add a sequence of contiguous textboxes going either to the right or down to an existing image
    Args:
        img: image to be edited
        pane_rectangle: first rectangle in the sequence
        texts: a list of strings where each item corresponds to an individual text box
        direction: right or down

    Returns:
        the edited image object
    """
    textbox_rectangle = get_textbox_rectangle_from_pane(pane_rectangle=pane_rectangle, texts=texts, direction=direction)
    next_coordinates = textbox_rectangle.top_left
    box_width, box_height = textbox_rectangle.dimensions.width_by_height
    for text in texts:
        next_rectangle = textbox_rectangle.from_coordinates(next_coordinates)
        img = add_text(img=img, rectangle=next_rectangle, text=text)

        if direction == 'right':
            next_coordinates.x += box_width
        elif direction == 'down':
            next_coordinates.y += box_height
        else:
            raise InvalidDirectionError(f'direction must be "right" or "down": {direction}')

    return img


def add_fret_indexes(img: Img, fret_index_rectangle: GeometricRectangle) -> Img:
    """add a sequence of fret indexes to the image"""
    fret_numbers: List[str] = [str(i) for i in range(1, 13)]
    return add_sequence_of_contiguous_textboxes(img=img, pane_rectangle=fret_index_rectangle, texts=fret_numbers,
                                                direction='right')


def add_gstring_indexes(img: Img, gstring_index_rectangle: GeometricRectangle, tuning: List[str]) -> Img:
    """"""
    numeral_index: List[str] = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    notes_with_num_index = [f'{num} - {note}' for num, note in zip(numeral_index, tuning)]
    return add_sequence_of_contiguous_textboxes(img=img, pane_rectangle=gstring_index_rectangle,
                                                texts=notes_with_num_index, direction='down')


def add_title_text(img: Img, title_pane: GeometricRectangle, key: str, mode: str, num_gstrings: int) -> Img:
    """add title to the graphic that outlines what scale is displayed in the graphic"""
    title = f'{key} {mode} on a {num_gstrings} string guitar'
    return add_text(img=img, rectangle=title_pane, text=title)


def add_fret_marks() -> Img:
    pass


def get_output_file_name(key: str, scale: str, num_gstrings: Union[str, int]) -> str:
    """return self documenting image file name"""
    key_alpha: str = key.replace('#', 'sharp')
    return f'{key_alpha}_{scale}_{num_gstrings}gstrings.png'


def main():
    clear_log()
    user_num_gstrings: int = input_num_gstrings()
    user_key: str = input_select_key(get_chromatic_notes('C'))
    user_scale: str = input_select_scale()
    num_neck_notes: int = len(get_chromatic_notes('C'))

    image_panes = ImagePanes(width=WIDTH, height=HEIGHT, num_strings=user_num_gstrings)
    img_dimensions = image_panes.get_img_dimensions()
    img = get_blank_img(width=img_dimensions.width, height=img_dimensions.height)

    fretboard_rectangle = image_panes.get_fretboard_pane_rectangle()
    fret_dimensions = get_fret_dimensions(fretboard_dimensions=fretboard_rectangle.dimensions,
                                          num_strings=user_num_gstrings, num_notes=num_neck_notes)
    gtuning_open = get_gtuning(num_gstrings=user_num_gstrings, gtuning_8strings=get_tuning_standard())
    gtuning_first_fret = get_notes_of_next_fret(gtuning_open, get_chromatic_notes('C'))
    chromatic_notes_in_selected_key = get_chromatic_notes(key=user_key)
    user_scale_notes = get_scale_notes(scale=user_scale, scale_chromatic=chromatic_notes_in_selected_key)

    first_fret = GeometricRectangle(top_left=fretboard_rectangle.top_left,
                                    dimensions=fret_dimensions)
    fretboard_struct = get_fretboard_struct(gtuning=gtuning_first_fret, first_fret=first_fret)
    img = populate_fretboard(img=img, fretboard_struct=fretboard_struct, user_scale_notes=user_scale_notes)

    img = add_fret_indexes(img=img, fret_index_rectangle=image_panes.get_fret_index_pane_rectangle())
    img = add_gstring_indexes(img=img, gstring_index_rectangle=image_panes.get_gstring_index_pane_rectangle(),
                              tuning=gtuning_open)
    img = add_title_text(img=img,
                         title_pane=image_panes.get_title_pane_rectangle(),
                         key=user_key,
                         mode=user_scale,
                         num_gstrings=user_num_gstrings)
    output_file = get_output_file_name(key=user_key, scale=user_scale, num_gstrings=user_num_gstrings)
    save_img(img, output_file)


if __name__ == '__main__':
    main()
