# Health and Diet Monitoring ChatBot

A personalized health assistant that helps users track their diet based on body type, weight, and health goals, plan and monitor workout routines, and receive helpful feedback to maintain a healthier lifestyle.

## 📋 Overview

This web application integrates an LLM-powered chatbot (using Ollama) to provide personalized health and fitness guidance. The chatbot analyzes user inputs about their body metrics, dietary preferences, and fitness goals to offer tailored recommendations and tracking capabilities.

## ✨ Features

- **Personalized Diet Tracking**: Monitor your nutrition based on your specific body type, weight, and health goals
- **Workout Planning**: Get customized workout routines and track your progress
- **Real-time Feedback**: Receive insights and suggestions to improve your health journey
- **User-friendly Interface**: Clean, responsive design built with Bootstrap for seamless experience across devices

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Django (Python)
- **AI Model**: Ollama integration for intelligent responses
- **Database**: SQLite3

## 📁 Project Structure

```
gym-final/
├── Project/                # Django project directory
│   ├── __init__.py
│   ├── settings.py         # Django settings
│   ├── urls.py             # Main URL configuration
│   └── wsgi.py
├── healthapp/              # Django application
│   ├── __init__.py
│   ├── admin.py
│   ├── models.py
│   ├── views.py            # View functions
│   ├── urls.py             # App-level URL routing
│   └── templates/          # HTML templates
│       └── index.html      # Main application interface
├── templates/              # Project-level templates
├── static/                 # Static files (CSS, JS, images)
├── manage.py               # Django management script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
└── db.sqlite3              # SQLite database
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Ollama installed and running locally

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/raytracer11/gym-final.git
   cd gym-final
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Ollama (if not already installed):
   Follow the instructions at [Ollama's official documentation](https://ollama.ai/download)

5. Run migrations:
   ```bash
   python manage.py migrate
   ```

6. Start the development server:
   ```bash
   python manage.py runserver
   ```

7. Access the application at `http://127.0.0.1:8000/`

## 💻 Usage

1. Register or log in to your account
2. Complete your profile with body metrics and health goals
3. Use the chatbot interface to:
   - Track your daily meals and nutritional intake
   - Generate workout plans based on your goals
   - Receive feedback on your progress
   - Ask health and fitness related questions

## 🧰 Configuration

The application uses Django's settings module for configuration. Key settings:

- **Ollama Integration**: Configure the Ollama model endpoint in `settings.py`
- **Database**: Default is SQLite3, can be changed in database settings
- **Static Files**: CSS, JavaScript, and images are stored in the static directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

GitHub: [@raytracer11](https://github.com/raytracer11)
