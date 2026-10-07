import tifffile

with tifffile.TiffFile("mnt/raid10/opendata_files/images_breslow_categ_final/breslow_1_2/f1f6t66q.tif") as tif:
    page = tif.pages[0]
    print("Размер:", page.shape)
    print("dtype:", page.dtype)
    print("Каналы:", page.samplesperpixel)
    print("Photometric:", page.photometric)
    print("Compression:", page.compression)
    print("Tiled:", page.is_tiled)
    print("Страниц всего:", len(tif.pages))
    if page.is_tiled:
        print("Tile size:", page.tilewidth, "x", page.tilelength)
