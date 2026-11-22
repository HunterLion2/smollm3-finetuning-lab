# SmolLM3 Fine-tuning Lab

**Çınar Özgürgil · LLM / PEFT / LoRA deney çalışması**

SmolLM3-3B'yi ölçülebilir bir görevde LoRA ile uyarlamak ve temel modelle aynı test verisinde karşılaştırmak için hazırladığım bağımsız çalışma. İlk görev BANKING77 metinlerinden destek niyeti sınıflandırmasıdır.

> **Durum: Gerçek SmolLM3 eğitimi henüz çalıştırılmadı.** Depo deney kodunu ve ölçüm yöntemini içerir; eğitilmiş model veya başarı artışı iddia etmez. [Durum kaydı](reports/status.json). TherapeiAİ'nin özel kodunu veya kullanıcı verisini içermez.

## İnceleme rotası

- [Yapılandırma](config.json): model, veri ve eğitim ayarları.
- [Deney kodu](lab.py): hazırlık → eğitim → ayrı test → karşılaştırma.
- [Veri kartı](DATA_CARD.md): kaynak, lisans, ayrım ve sınırlar.
- [Sonuç durumu](reports/README.md): önce/sonra ölçüm planı.
- [Kontroller](tests/test_lab.py): veri sızıntısı ve yanlış karşılaştırma denetimleri.

## Model ve veri

| Seçim | Değer |
| --- | --- |
| Model | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), Apache-2.0 |
| Veri | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), CC BY 4.0, İngilizce / 77 niyet |
| Uyarlama | PEFT/LoRA, q/k/v/o projection katmanları |
| Train | Her sınıftan 20 örnek; 1.540 toplam |
| Validation | Orijinal train bölmesinden ayrı 5 örnek/sınıf; 385 toplam |
| Test | Orijinal test bölmesinden 10 örnek/sınıf; 770 toplam |
| Ölçüt | Tam etiket doğruluğu ve geçersiz çıktı oranı |

Hazırlama adımı model/veri dallarını değişmez commit SHA'larına çözer; örnek kimliklerini, kod hash'ini ve veri hash'ini kaydeder. Sonraki adımlar bu snapshot'ı kullanır. Başka gün yeniden hazırlamak farklı kaynak sürümü seçebilir; aynı deneyi tekrarlamak için snapshot ve manifest korunmalıdır.

## Çalıştırma

Python **3.11** ve CUDA destekli NVIDIA GPU gerekir. Nicemleme kullanılmayan bu tarif için 24 GB VRAM başlangıç hedefidir; gerçek bellek gereksinimi henüz ölçülmedi. Komutlar ücretli GPU işi başlatmaz ve Hub'a otomatik yükleme yapmaz.

```sh
git clone https://github.com/HunterLion2/smollm3-finetuning-lab.git
cd smollm3-finetuning-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Veri bölmelerini ve değişmez kaynak revizyonlarını hazırla:
python lab.py prepare

# Temel modeli eğitim öncesi aynı test kümesinde ölç:
python lab.py evaluate --mode base

# Yalnızca train ile eğit; validation üzerinde loss izle:
python lab.py train

# Adaptörü aynı test örnekleriyle değerlendir:
python lab.py evaluate --mode adapter
python lab.py compare
```

Windows PowerShell sanal ortam komutu: `.venv\Scripts\Activate.ps1`. PyTorch kurulumu CUDA ortamıyla eşleşmelidir. GPU yoksa eğitim/üretim komutları açık hata verir.

## Ayarlar ve çıktı

LoRA r=16, alpha=32, dropout=0.05; learning rate=2e-4; epoch=1; batch=1; gradient accumulation=8; max length=2048; seed=42. Gradient checkpointing açık; destek varsa BF16, yoksa FP16 kullanılır. TRL prompt-completion biçiminde yalnızca completion token'ları üzerinde loss hesaplar.

Başlangıç ayarlarıdır; en iyi parametreler oldukları iddia edilmez. Test verisi eğitim veya validation için kullanılmaz. Normalize edilmiş metin tekrarı yakalanır; uzun örnekler sessizce kırpılmaz.

| Dosya | İçerik |
| --- | --- |
| `data/experiment.json` | Kaynak revizyonları, veri snapshot'ı ve hash |
| `outputs/adapter/` | LoRA ağırlıkları, tokenizer ve deney kimliği |
| `runs/training.json` | Gerçek eğitim, donanım ve ortam kaydı |
| `runs/pip-freeze.txt` | Çalıştırmada kullanılan tam paket listesi |
| `runs/base.json`, `runs/adapter.json` | Her test girdisi, hedef ve ham çıktı |
| `runs/comparison.md` | Doğruluk farkı ve ilk 10 sabit örnek |

İndirilen veri ve büyük çıktılar Git tarafından izlenmez. Sonuçları inceleyip paylaşılabilir raporları `reports/` içine ekleyin; durum kaydını ancak gerçek ölçümlerden sonra güncelleyin. Önceki model/ölçüm dosyaları sessizce ezilmez.

## Kontroller

```sh
python -m unittest discover -s tests -v
python -m py_compile lab.py
python lab.py --help
# requirements kuruluysa:
python tests/cpu_smoke.py
```

GitHub Actions veri/raporlama birim kontrollerini ve küçük rastgele modelle bir CPU eğitim-adaptör yükleme kontrolünü çalıştırır. **Sentetik CPU kontrolü, SmolLM3 eğitimi veya kalite ölçümü değildir.**

## Sınırlar

Küçük örneklem, tek seed ve tek görev. Türkçe sohbet, genel akıl yürütme veya klinik kullanım için kanıt değildir. Sonuç iyileşmezse rapor bunu açıkça yazar. Eksik sonuçlardan veya uyumsuz deneylerden başarı raporu üretmez. Teste tekrar tekrar bakarak parametre ayarlamak test kümesine aşırı uyum doğurur.

[TRL v0.24 SFTTrainer](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer) · [PEFT](https://huggingface.co/docs/peft/index) · [TherapeiAİ teknik vitrini](https://github.com/HunterLion2/HunterLion2/blob/main/projects/therapeiai.md)
