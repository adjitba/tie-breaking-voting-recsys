# Hasil uji kandidat terhadap kriteria yang ditetapkan sebelumnya

Kriteria: pada k = 30, verdict 'berbeda bermakna' dan positif terhadap Plurality, Approval, dan rating sum,
untuk P@10 DAN NDCG@10, di minimal 2 dari 3 dataset.

## shrunk_plurality: memenuhi di 0 dari 3 dataset -> TIDAK BERHASIL

| Dataset | Metrik | Pembanding | Kandidat | Pembanding | Selisih relatif | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| ML-100K | P@10 | plurality | 0.0675 | 0.0817 | -17.3% | berbeda bermakna |
| ML-100K | P@10 | approval | 0.0675 | 0.0796 | -15.2% | berbeda bermakna |
| ML-100K | P@10 | rating_sum | 0.0675 | 0.0791 | -14.6% | berbeda bermakna |
| ML-100K | NDCG@10 | plurality | 0.0868 | 0.1159 | -25.1% | berbeda bermakna |
| ML-100K | NDCG@10 | approval | 0.0868 | 0.1145 | -24.2% | berbeda bermakna |
| ML-100K | NDCG@10 | rating_sum | 0.0868 | 0.1154 | -24.8% | berbeda bermakna |
| FilmTrust | P@10 | plurality | 0.1356 | 0.1532 | -11.5% | berbeda bermakna |
| FilmTrust | P@10 | approval | 0.1356 | 0.1525 | -11.1% | berbeda bermakna |
| FilmTrust | P@10 | rating_sum | 0.1356 | 0.1598 | -15.2% | berbeda bermakna |
| FilmTrust | NDCG@10 | plurality | 0.3395 | 0.4245 | -20.0% | berbeda bermakna |
| FilmTrust | NDCG@10 | approval | 0.3395 | 0.4123 | -17.7% | berbeda bermakna |
| FilmTrust | NDCG@10 | rating_sum | 0.3395 | 0.4315 | -21.3% | berbeda bermakna |
| Amazon | P@10 | plurality | 0.0236 | 0.0255 | -7.7% | berbeda bermakna |
| Amazon | P@10 | approval | 0.0236 | 0.0254 | -7.3% | berbeda bermakna |
| Amazon | P@10 | rating_sum | 0.0236 | 0.0259 | -9.1% | berbeda bermakna |
| Amazon | NDCG@10 | plurality | 0.0770 | 0.0833 | -7.6% | berbeda bermakna |
| Amazon | NDCG@10 | approval | 0.0770 | 0.0830 | -7.2% | berbeda bermakna |
| Amazon | NDCG@10 | rating_sum | 0.0770 | 0.0845 | -8.9% | berbeda bermakna |

## shrunk_approval: memenuhi di 0 dari 3 dataset -> TIDAK BERHASIL

| Dataset | Metrik | Pembanding | Kandidat | Pembanding | Selisih relatif | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| ML-100K | P@10 | plurality | 0.0627 | 0.0817 | -23.3% | berbeda bermakna |
| ML-100K | P@10 | approval | 0.0627 | 0.0796 | -21.3% | berbeda bermakna |
| ML-100K | P@10 | rating_sum | 0.0627 | 0.0791 | -20.8% | berbeda bermakna |
| ML-100K | NDCG@10 | plurality | 0.0798 | 0.1159 | -31.2% | berbeda bermakna |
| ML-100K | NDCG@10 | approval | 0.0798 | 0.1145 | -30.3% | berbeda bermakna |
| ML-100K | NDCG@10 | rating_sum | 0.0798 | 0.1154 | -30.8% | berbeda bermakna |
| FilmTrust | P@10 | plurality | 0.1154 | 0.1532 | -24.7% | berbeda bermakna |
| FilmTrust | P@10 | approval | 0.1154 | 0.1525 | -24.4% | berbeda bermakna |
| FilmTrust | P@10 | rating_sum | 0.1154 | 0.1598 | -27.8% | berbeda bermakna |
| FilmTrust | NDCG@10 | plurality | 0.2711 | 0.4245 | -36.1% | berbeda bermakna |
| FilmTrust | NDCG@10 | approval | 0.2711 | 0.4123 | -34.2% | berbeda bermakna |
| FilmTrust | NDCG@10 | rating_sum | 0.2711 | 0.4315 | -37.2% | berbeda bermakna |
| Amazon | P@10 | plurality | 0.0235 | 0.0255 | -7.9% | berbeda bermakna |
| Amazon | P@10 | approval | 0.0235 | 0.0254 | -7.5% | berbeda bermakna |
| Amazon | P@10 | rating_sum | 0.0235 | 0.0259 | -9.3% | berbeda bermakna |
| Amazon | NDCG@10 | plurality | 0.0753 | 0.0833 | -9.6% | berbeda bermakna |
| Amazon | NDCG@10 | approval | 0.0753 | 0.0830 | -9.2% | berbeda bermakna |
| Amazon | NDCG@10 | rating_sum | 0.0753 | 0.0845 | -10.8% | berbeda bermakna |

## quality_approval: memenuhi di 0 dari 3 dataset -> TIDAK BERHASIL

| Dataset | Metrik | Pembanding | Kandidat | Pembanding | Selisih relatif | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| ML-100K | P@10 | plurality | 0.0798 | 0.0817 | -2.3% | tidak konklusif |
| ML-100K | P@10 | approval | 0.0798 | 0.0796 | +0.2% | setara |
| ML-100K | P@10 | rating_sum | 0.0798 | 0.0791 | +0.9% | setara |
| ML-100K | NDCG@10 | plurality | 0.1132 | 0.1159 | -2.3% | tidak konklusif |
| ML-100K | NDCG@10 | approval | 0.1132 | 0.1145 | -1.1% | beda kecil, setara secara praktis |
| ML-100K | NDCG@10 | rating_sum | 0.1132 | 0.1154 | -1.9% | setara |
| FilmTrust | P@10 | plurality | 0.1497 | 0.1532 | -2.3% | beda kecil, setara secara praktis |
| FilmTrust | P@10 | approval | 0.1497 | 0.1525 | -1.9% | beda kecil, setara secara praktis |
| FilmTrust | P@10 | rating_sum | 0.1497 | 0.1598 | -6.3% | berbeda bermakna |
| FilmTrust | NDCG@10 | plurality | 0.4048 | 0.4245 | -4.6% | tidak konklusif |
| FilmTrust | NDCG@10 | approval | 0.4048 | 0.4123 | -1.8% | beda kecil, setara secara praktis |
| FilmTrust | NDCG@10 | rating_sum | 0.4048 | 0.4315 | -6.2% | berbeda bermakna |
| Amazon | P@10 | plurality | 0.0254 | 0.0255 | -0.4% | setara |
| Amazon | P@10 | approval | 0.0254 | 0.0254 | +0.1% | setara |
| Amazon | P@10 | rating_sum | 0.0254 | 0.0259 | -1.9% | beda kecil, setara secara praktis |
| Amazon | NDCG@10 | plurality | 0.0825 | 0.0833 | -1.0% | setara |
| Amazon | NDCG@10 | approval | 0.0825 | 0.0830 | -0.6% | beda kecil, setara secara praktis |
| Amazon | NDCG@10 | rating_sum | 0.0825 | 0.0845 | -2.4% | beda kecil, setara secara praktis |

## Semua metode, k = 30, @10

| Dataset | Metode | P@10 | NDCG@10 | POP@10 |
| --- | --- | --- | --- | --- |
| Amazon | approval | 0.0254 | 0.0830 | 43.1 |
| Amazon | borda | 0.0202 | 0.0672 | 41.0 |
| Amazon | copeland | 0.0200 | 0.0668 | 40.8 |
| Amazon | least_misery | 0.0200 | 0.0657 | 38.9 |
| Amazon | most_pleasure | 0.0203 | 0.0663 | 38.7 |
| Amazon | plurality | 0.0255 | 0.0833 | 41.3 |
| Amazon | quality_approval | 0.0254 | 0.0825 | 42.7 |
| Amazon | rating_sum | 0.0259 | 0.0845 | 43.2 |
| Amazon | shrunk_approval | 0.0235 | 0.0753 | 42.1 |
| Amazon | shrunk_plurality | 0.0236 | 0.0770 | 40.0 |
| FilmTrust | approval | 0.1525 | 0.4123 | 478.9 |
| FilmTrust | borda | 0.1402 | 0.3598 | 459.4 |
| FilmTrust | copeland | 0.1152 | 0.2629 | 368.2 |
| FilmTrust | least_misery | 0.0891 | 0.1520 | 271.4 |
| FilmTrust | most_pleasure | 0.1148 | 0.2379 | 378.2 |
| FilmTrust | plurality | 0.1532 | 0.4245 | 466.2 |
| FilmTrust | quality_approval | 0.1497 | 0.4048 | 470.0 |
| FilmTrust | rating_sum | 0.1598 | 0.4315 | 508.6 |
| FilmTrust | shrunk_approval | 0.1154 | 0.2711 | 352.4 |
| FilmTrust | shrunk_plurality | 0.1356 | 0.3395 | 391.1 |
| ML-100K | approval | 0.0796 | 0.1145 | 285.2 |
| ML-100K | borda | 0.0606 | 0.0828 | 263.6 |
| ML-100K | copeland | 0.0550 | 0.0714 | 208.5 |
| ML-100K | least_misery | 0.0410 | 0.0440 | 107.4 |
| ML-100K | most_pleasure | 0.0358 | 0.0399 | 145.3 |
| ML-100K | plurality | 0.0817 | 0.1159 | 233.3 |
| ML-100K | quality_approval | 0.0798 | 0.1132 | 278.3 |
| ML-100K | rating_sum | 0.0791 | 0.1154 | 297.8 |
| ML-100K | shrunk_approval | 0.0627 | 0.0798 | 203.6 |
| ML-100K | shrunk_plurality | 0.0675 | 0.0868 | 167.9 |
