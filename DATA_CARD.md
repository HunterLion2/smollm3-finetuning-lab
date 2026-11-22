# Veri kartı — BANKING77

Kaynak: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77). Sağlayıcı: PolyAI. Lisans: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). İlgili çalışma: Casanueva ve diğerleri, [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807).

Orijinal İngilizce metinler 77 niyetten tek etiket üretme görevine dönüştürülür. Hazırlama komutu veriyi resmî kaynaktan indirir ve revizyonunu kaydeder. Prompt'a etiket listesi eklenir; hedef yalnızca doğru etiket adıdır.

Önce orijinal test bölmesi korunur. Büyük/küçük harf ve boşluk normalizasyonuyla tekrarlar elenir; testte bulunan metinler eğitim havuzundan çıkarılır. Orijinal train bölmesi seed ile sınıf bazında train/validation olarak ayrılır. Beklenen sayılar 1.540 / 385 / 770'tir. Yeterli benzersiz örnek yoksa hata verilir; sayı sessizce azaltılmaz. Exact duplicate kontrolü anlamsal benzerliğin tamamını yakalamaz.

Bu veri Türkçe duygu analizi veya ruh sağlığı eğitimi verisi değildir; TherapeiAİ'nin klinik/ürün başarısına kanıt olarak kullanılmaz. Kullanıcının banka hesabı veya kişisel mali kayıtları kullanılmaz.

Yerel veri snapshot'ını ve manifest'i yeniden üretim için koruyun. Başka veri eklendiğinde kaynak, lisans, izinler ve kişisel veri durumu ayrıca belgelenmelidir.
