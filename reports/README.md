# Sonuç durumu

**Gerçek SmolLM3 eğitimi veya model değerlendirmesi henüz çalıştırılmadı.**

| Ölçüt | Temel model | LoRA |
| --- | --- | --- |
| Test doğruluğu | Ölçülmedi | Ölçülmedi |
| Geçersiz çıktı oranı | Ölçülmedi | Ölçülmedi |
| Eğitim süresi / GPU belleği | Uygulanmaz | Ölçülmedi |

[Durum kaydı](status.json). Ölçülmemiş değerler 0 yerine null tutulur.

`python lab.py compare` yalnızca iki gerçek değerlendirme mevcutsa çalışır. Model/veri revizyonları, bağımlılık sürümleri, üretim ayarları ve test örnekleri eşleşmezse hata verir. Sonuç iyi, eşit veya kötü olabilir; olumsuz sonuç da korunur.

Rapor sabit test sırasındaki ilk 10 örneği gösterir; yalnızca başarılı çıktılar seçilmez. Tüm ham çıktılar `runs/*.json` içinde bulunur. Gerçek rapor incelendikten sonra bu dizine eklenmelidir; eğitim kaydı, GPU, süre, paket sürümleri ve veri manifest'iyle birlikte paylaşılmalıdır.

Hata analizinde karışan sınıflar, geçersiz/uzun yanıtlar ve temel modele göre kötüleşen örnekler incelenecek. Küçük örneklem ve tek seed sınırlamaları yazılacak. Burada uydurulmuş model çıktısı veya başarı oranı yoktur.
