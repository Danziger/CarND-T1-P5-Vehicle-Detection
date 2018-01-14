import src.helpers.windows as WIN


COLOR_SPACES = [
    "RGB",
    "HSV",
    "LUV",
    "HLS",
    "YUV",
    "YCR_CB", # YCrCb,
    "GRAY"
]


def WINDOWS(WIDTH):
    # XS - 64 px
    windows_xs = WIN.slide_window(0, WIDTH, 400, 528, (64, 64), (0.5, 0.5))

    # S - 96 px
    windows_s = WIN.slide_window(0, WIDTH, 400, 544, (96, 96))

    # M - 128 px
    windows_m = WIN.slide_window(0, WIDTH, 400, 592, (128, 128))
    
    return windows_xs + windows_s + windows_m
