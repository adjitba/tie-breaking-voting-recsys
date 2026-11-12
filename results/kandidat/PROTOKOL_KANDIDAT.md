# Protokol uji kandidat aturan voting baru (ditetapkan sebelum eksperimen dijalankan)

Tanggal: 4 November 2026. Ditulis sebelum hasil kandidat dilihat.

## Kandidat (tiga, tidak ditambah setelah hasil keluar)

Notasi per user target u: V(i) = suara Plurality (dibagi rata bila favorit seri), A(i) = jumlah tetangga dengan rating >= tau,
n_i = jumlah tetangga yang merating i, s_i = jumlah rating, m = rata-rata n_i atas kandidat,
B(i) = (s_i + m mu_C)/(n_i + m), b(i) = (B(i) - r_min)/Delta.

1. Shrunk Plurality:  SP(i) = (V(i) + m * p) / (n_i + m),  p = sum_i V(i) / sum_i n_i
2. Shrunk Approval:   SA(i) = (A(i) + m * q) / (n_i + m),  q = sum_i A(i) / sum_i n_i
3. Quality-weighted Approval: QA(i) = A(i) * b(i)

## Pembanding

Plurality, Approval, Borda, Copeland, Least Misery, Most Pleasure (pemecahan seri acak, nilai harapan pasti),
dan rating sum (jumlah rating tetangga, skor user-kNN klasik tanpa bobot).

## Pengaturan

ML-100K, FilmTrust, Amazon; k = 5, 10, 15, 20, 30; N = 5, 10, 20, 50; protokol A; 5 seed (0-4), split sama
dengan eksperimen Masthoff sehingga perbandingan berpasangan per user.

## Kriteria berhasil (untuk satu kandidat)

Pada k = 30, kandidat menghasilkan verdict "berbeda bermakna" dan positif (TOST margin 5%, CI 90%)
terhadap SETIAP pembanding Plurality, Approval, dan rating sum, untuk P@10 DAN NDCG@10,
di minimal 2 dari 3 dataset.

Bila tidak ada kandidat yang memenuhi kriteria, naskah kembali ke bingkai kerangka pemecahan seri B(i).
Bila ada yang memenuhi, kandidat itu harus dikonfirmasi pada dataset baru yang belum pernah dipakai.
