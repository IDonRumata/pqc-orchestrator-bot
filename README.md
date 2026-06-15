# PQC Strategic Orchestrator Bot

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Multi%2DAgent-0088cc?logo=telegram)](https://core.telegram.org/bots/api)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20API-FF6C37?logo=openrouter&logoColor=white)](https://openrouter.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Асинхронный мультиагентный Telegram-бот для стратегии, комплаенса и финансов стартапа в сфере постквантовой криптографии (PQC) и кибербезопасности. Интегрирует пять специализированных AI-агентов, RAG базу знаний (EU законы, гранты) и мониторинг регуляторных изменений.

**Целевая аудитория:** Основатели PQC стартапов (ЕС, Польша)  
**Язык:** Русский (интерфейс) + многоязычные агенты  
**Статус:** Личный инструмент (в разработке)

---

## 🧠 Мультиагентная архитектура

### Пять специализированных агентов:

| Агент | Специализация | Вызывается когда |
|---|---|---|
| 💰 **Financial Director** | Бюджет, финансирование, гранты, инвестиции | "Сколько денег нам нужно?", "Какие EU гранты для PQC?" |
| 🔬 **Cryptography Scientist** | ML-KEM, ML-DSA, стандарты NIST, техчасть | "Как мигрировать на ML-DSA?", "Какие гибридные схемы?" |
| ⚖️ **EU Compliance Lawyer** | DORA, NIS2, GDPR, регуляция | "Что требует DORA?", "Какие штрафы за нарушения?" |
| 🎁 **EU Grants Expert** | Horizon Europe, PNRR, локальные гранты (Польша) | "Есть ли гранты для нашей идеи?", "Как писать заявку?" |
| 🧐 **Chief Critic** | Контрольная проверка, риск-ассессмент, критика идей | "А что может пойти не так?", "Проверь мою стратегию" |

---

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────┐
│ User Query (Telegram)                                   │
│ "Какие EU гранты для PQC стартапа в Польше?"           │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Intent Router (Claude)                                   │
│ • Определяет релевантные агенты (1-3 из 5)              │
│ • Анализирует контекст (стартап, страна, бюджет)        │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ RAG Knowledge Base (PostgreSQL + pgvector)               │
│ • EU законы (DORA, NIS2, GDPR)                          │
│ • EU гранты (Horizon Europe, PNRR)                      │
│ • PQC стандарты (NIST FIPS 203/204/205)                 │
│ • Локальные гранты (Польша, Чехия)                      │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Agent Selection & Execution                              │
│ Router выбирает агентов (параллельное выполнение):       │
│ • EU Grants Expert (primary) → поиск грантов             │
│ • Financial Director (secondary) → бюджет               │
│ • Compliance Lawyer (context) → регуляция                │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Response Synthesis                                       │
│ • Chief Critic проверяет полноту                         │
│ • Объединение ответов от агентов                        │
│ • Форматирование для Telegram                           │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Background Monitoring (Cron)                             │
│ • Каждые 6 часов: скан новостей (EU законы, гранты)    │
│ • Уведомления при критических изменениях                │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ Ключевые возможности

| Функция | Описание |
|---------|---------|
| 🤖 **Мультиагентный роутинг** | Автоматически выбирает 1–3 агента под вопрос |
| 📚 **RAG база знаний** | Векторная БД с EU законами, грантами, стандартами PQC |
| 🔔 **Мониторинг регуляции** | Автоматический скан новостей → алерты при изменениях (DORA, NIS2) |
| 💬 **Асинхронное выполнение** | Параллельные запросы к агентам (быстро) |
| 🧐 **Критическая оценка** | Chief Critic проверяет полноту ответов |
| 📊 **Контекст памяти** | Помнит историю разговора (многовход. диалоги) |

---

## 🛠️ Стек технологий

**Core:**
- Python 3.11+ (async/await)
- aiogram 3.x (Telegram Bot API)
- PostgreSQL 15+ с pgvector (RAG)

**LLM & Embeddings:**
- OpenRouter API (Claude, GPT, LLaMA)
- fastembed (локальные эмбеддинги, бесплатно)
- OpenAI embeddings (опционально, платно)

**Async & Scheduling:**
- APScheduler (cron для мониторинга)
- asyncio (параллельное выполнение агентов)

**Deployment:**
- Docker + Docker Compose
- systemd service (на VPS)

---

## 🚀 Быстрый старт

### Docker (рекомендуемо)

```bash
git clone https://github.com/IDonRumata/pqc-orchestrator-bot.git
cd pqc-orchestrator-bot

cp .env.example .env
nano .env  # Заполни TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY, POSTGRES_PASSWORD

docker compose up -d --build
docker compose logs -f bot
```

### Локально (без Docker)

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# или с uv:
uv venv && uv pip install -r requirements.txt

cp .env.example .env
# Подними PostgreSQL вручную + pgvector
python main.py
```

### Наполнение базы знаний

```bash
# Положи файлы (PDF, markdown) в knowledge_base/
cp ~/eu_laws/*.pdf knowledge_base/legal/

# Ингест в БД
docker compose exec bot python -m src.ingest \
  --path knowledge_base \
  --source legal

# или для грантов:
python -m src.ingest --path knowledge_base --source grants
```

---

## ⚙️ Конфигурация

| Переменная | Пример | Где получить |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `123456:ABCdef...` | [@BotFather](https://t.me/botfather) |
| `ALLOWED_USER_IDS` | `987654321,111222333` | [@userinfobot](https://t.me/userinfobot) |
| `OPENROUTER_API_KEY` | `sk-or-...` | [openrouter.ai](https://openrouter.ai/keys) |
| `DATABASE_URL` | `postgresql://user:pass@localhost/pqc` | Твой PostgreSQL |
| `POSTGRES_PASSWORD` | `strong_password` | Выбираешь сам |
| `OPENAI_API_KEY` | (optional) | [openai.com](https://platform.openai.com) — для качественных эмбеддингов |

---

## 🤖 Команды бота

| Команда | Действие |
|---|---|
| `/start` | Начало, инструкция |
| `/ask` | Спросить (обычный вопрос) |
| `/grant` | Поиск грантов EU для стартапа |
| `/compliance` | Вопрос по регуляции (DORA/NIS2) |
| `/crypto` | Техвопрос про PQC/ML-KEM |
| `/finance` | Финансовая консультация |
| `/check` | Критическая оценка идеи |
| `/monitor` | Включить/выключить мониторинг новостей |
| `/knowledge` | Какие источники в базе знаний |

---

## 💡 Примеры использования

### Вопрос про гранты

```
Юзер: /grant
      Страна: Польша
      Стадия: pre-seed
      Фокус: PQC стартап

Бот: 🎁 EU гранты для тебя:

1. **Horizon Europe – DIGITAL-2024-CRYPTOSECURITY**
   Бюджет: €100k–500k на проект
   Deadline: 15 сентября 2024
   Fit: Очень хорошо (70%)
   
2. **PNRR – Cyberattacks Protection (Польша)**
   Бюджет: €50k–300k
   Fit: Хорошо (60%)
   
3. **NCBiR – TECHNIKON (Польша)**
   Бюджет: €150k–2M
   Fit: Отлично (85%)

📝 Совет: Начни с PNRR (быстро) или NCBiR (больше денег).
```

### Критическая оценка стратегии

```
Юзер: /check
      Моя стратегия: Создать ML-DSA гибридный TLS прокси для финбанков

Бот: 🧐 Chief Critic анализирует...

✅ Сильные стороны:
   • Большой рынок (финбанки платят за security)
   • EU требует PQC to 2026 (DORA) → спрос есть
   • Гибридный подход = минимальный risk на legacy системы

⚠️ Риски:
   • Конкуренция: Google, Apple уже работают на гибридах
   • Regulatory lag: DORA даёт 3 года → может быть late to market
   • Technical risk: ML-DSA стандартов завершили только в 2024

💡 Рекомендация:
   → Позиционируй как "консалтинг + реализация" (не продукт)
   → Целевой сегмент: немецкие/польские банки (more regulations = more budget)
   → Timeline: 18–24 месяца до первого клиента
```

### Мониторинг DORA изменений

```
Bot (автоматически, фоновый сканер):
   
🚨 КРИТИЧНОЕ ОБНОВЛЕНИЕ – DORA

Европейский банк объявил новые требования к TLS версии
для критичной инфраструктуры (в силу 31 декабря 2026):

"All system interfaces must support TLS 1.3 minimum and 
consider quantum-safe key exchanges (Post-Quantum Cryptography)..."

👉 Это делает твой гибридный TLS прокси ещё более актуальным!
   Рекомендуем обновить sales pitch и pitch deck.
```

---

## 📊 Примерный workflow день в день

```
9:00 AM: Я просыпаюсь
→ /grant "Какие гранты для PQC в Чехии?"
→ Бот: EU Grants Expert + Financial Director отвечают (30 сек)

10:00 AM: Я готовлю план миграции
→ /crypto "Какой порядок миграции RSA → ML-DSA?"
→ Бот: Cryptography Scientist + Chief Critic (1 мин)

12:00 PM: Обеденный перерыв
→ Бот в фоне скан новостей (всякие DORA изменения)
→ Если что-то критичное → алерт в TG

14:00 PM: Я проверяю финансовый план
→ /finance "Если мы привлечём $500k, на что нам хватит?"
→ Бот: Financial Director + Compliance Lawyer (1 мин)

18:00 PM: Я проверяю свою идею перед инвесторами
→ /check "Моя pitch deck — критика?"
→ Бот: Chief Critic полностью разбирает (2 мин)
```

---

## 🔐 Безопасность & Приватность

- ✅ Все данные локальны (в твоей БД)
- ✅ Telegram-only доступ (по ID)
- ✅ Логирование минимально, без сохранения истории
- ✅ GDPR compliant (твоя БД = твои данные)

---

## 📈 Будущие расширения

- [ ] Интеграция с GitHub (анализ кода на уязвимости)
- [ ] Sync с Google Docs / Notion (для документов стартапа)
- [ ] Web UI (вместо только Telegram)
- [ ] Multi-language (сейчас только Russian)
- [ ] Team sharing (сейчас личный инструмент)

---

## 📞 Контакты

- **Telegram:** [@DonRumataE](https://t.me/DonRumataE)
- **Email:** [andrei.maroz.eu@gmail.com](mailto:andrei.maroz.eu@gmail.com)

---

*Персональный мультиагентный conseil для PQC стартапа. Пять агентов, одна Telegram-группа.*
