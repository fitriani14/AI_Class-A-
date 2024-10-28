# Data input
x = [5, 5, 10, 3, 2, 6, 7]
y1 = 4
y2 = 2

# Fungsi untuk mencari indeks y1 atau y2
def Cari_index(x, y1, y2):
    if y1 in x:
        return x.index(y1)
    elif y2 in x:
        return x.index(y2)
    else:
        return 0

# Menjalankan fungsi dan mencetak output
hasil = Cari_index(x, y1, y2)
print("Output:", hasil)