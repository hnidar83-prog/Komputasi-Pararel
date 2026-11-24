from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import multiprocessing as mp


# Fungsi untuk mengubah sebagian gambar menjadi grayscale
def proses_grayscale(data):
    gambar = np.array(data)

    r = gambar[:, :, 0]
    g = gambar[:, :, 1]
    b = gambar[:, :, 2]

    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    hasil = np.stack((gray, gray, gray), axis=2)
    return hasil


if __name__ == "__main__":
    nama_file = "bunga.jpg"

    img = Image.open(nama_file).convert("RGB")

    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.title("Gambar Asli")
    plt.axis("off")
    plt.show()

    print("Nama file:", nama_file)
    print("Ukuran gambar:", img.size)

    # Membuka gambar
    img_serial = Image.open(nama_file).convert("RGB")

    width, height = img_serial.size
    pixels = img_serial.load()

    # Memulai pengukuran waktu
    start = time.perf_counter()

    # Proses gambar secara berurutan (Serial)
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            # Mengubah RGB menjadi grayscale
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)

            pixels[x, y] = (gray, gray, gray)

    # Menghitung waktu proses
    end = time.perf_counter()
    waktu_serial = end - start

    # Menampilkan hasil
    plt.figure(figsize=(8, 5))
    plt.imshow(img_serial)
    plt.title("Hasil Grayscale - Serial")
    plt.axis("off")
    plt.show()

    print("Pengolahan Serial Berhasil")
    print("Ukuran gambar:", width, "x", height)
    print("Waktu serial:", waktu_serial, "detik")

    # Membaca gambar
    gambar = np.array(Image.open(nama_file).convert("RGB"))
    tinggi, lebar, _ = gambar.shape

    # Membagi gambar menjadi 4 bagian
    bagian = np.array_split(gambar, 4, axis=0)

    # Mulai menghitung waktu
    start = time.perf_counter()

    # Proses 4 bagian secara paralel
    metode_mp = "fork" if "fork" in mp.get_all_start_methods() else None
    ctx = mp.get_context(metode_mp) if metode_mp else mp.get_context()
    with ctx.Pool(4) as pool:
        hasil = pool.map(proses_grayscale, bagian)

    # Menggabungkan kembali hasilnya
    gambar_hasil = np.concatenate(hasil, axis=0)
    img_paralel = Image.fromarray(gambar_hasil)

    # Menghitung waktu
    waktu_paralel = time.perf_counter() - start

    # Menampilkan hasil
    plt.figure(figsize=(8, 5))
    plt.imshow(img_paralel)
    plt.title("Hasil Grayscale - Paralel (4 Proses)")
    plt.axis("off")
    plt.show()

    print("Pengolahan Paralel Berhasil")
    print("Jumlah proses:", 4)
    print("Ukuran gambar:", lebar, "x", tinggi)
    print("Waktu paralel:", waktu_paralel, "detik")

    print("===== PERBANDINGAN WAKTU =====")
    print("Waktu Serial   :", waktu_serial, "detik")
    print("Waktu Paralel :", waktu_paralel, "detik")

    speedup = waktu_serial / waktu_paralel

    print("Speedup       :", speedup, "x")

    if waktu_paralel < waktu_serial:
        print("Kesimpulan: Proses paralel lebih cepat.")
    else:
        print("Kesimpulan: Proses serial lebih cepat.")