from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from datetime import date, datetime
import json
import groq
import uuid
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib import messages
from .models import UserProfile, DailyLog, Meal, Exercise, ChatMessage, ChatHistory, ServerSession
from django.conf import settings
from openai import OpenAI

import logging
logger = logging.getLogger(__name__)

# Generate a unique session ID for this server start
SERVER_SESSION_ID = str(uuid.uuid4())

# Create or get the current server session
try:
    session, created = ServerSession.objects.get_or_create(
        session_id=SERVER_SESSION_ID,
        defaults={'session_id': SERVER_SESSION_ID}
    )
    logger.info(f"Server session initialized: {SERVER_SESSION_ID}")
except Exception as e:
    logger.error(f"Error initializing server session: {e}")
    
def get_groq_client():
    return groq.Groq(api_key=settings.GROQ_API_KEY)

@login_required
def dashboard(request):
    # Get or create UserProfile
    try:
        user_profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        # Create default profile if it doesn't exist
        user_profile = UserProfile.objects.create(
            user=request.user,
            height=170,  # Default height
            weight=70,   # Default weight
            gender='male',  # Default gender
            body_type='mesomorph',
            activity_level='moderate',
            fitness_goal='maintenance'
        )
        return redirect('profile')  # Redirect to profile to complete setup
    
    today = date.today()
    
    # Get or create daily log
    daily_log, created = DailyLog.objects.get_or_create(
        user=request.user,
        date=today,
        defaults={'weight': user_profile.weight, 'water_consumed': 0, 'calories_consumed': 0, 'protein_consumed': 0}
    )
    
    # Check if this log has been processed in the current server session
    log_session_key = f"session_{SERVER_SESSION_ID}"
    if not hasattr(daily_log, log_session_key) or not getattr(daily_log, log_session_key, False):
        # This is first access in this server session, reset water
        logger.info(f"Resetting water for user {request.user.username} in session {SERVER_SESSION_ID}")
        daily_log.water_consumed = 0
        
        # Mark as processed in this session using request.session
        request.session[f"daily_log_{daily_log.id}_processed"] = True
        daily_log.save()
    
    # Get today's data
    meals = Meal.objects.filter(user=request.user, date=today).order_by('time')
    exercises = Exercise.objects.filter(user=request.user, date=today)
    
    # Calculate total carbs and fats from meals
    total_carbs = sum(meal.carbs for meal in meals)
    total_fats = sum(meal.fats for meal in meals)
    
    # Calculate target macros based on daily calorie needs
    daily_calories = user_profile.daily_calorie_needs
    carbs_target = (daily_calories * 0.5) / 4  # 50% of calories from carbs, 4 calories per gram
    fats_target = (daily_calories * 0.25) / 9  # 25% of calories from fats, 9 calories per gram
    
    # If no exercises for today or profile has been updated, create a new workout
    if not exercises.exists() or (hasattr(daily_log, 'last_updated') and daily_log.last_updated < user_profile.updated_at):
        exercises = create_default_workout(
            request.user, 
            today, 
            user_profile.fitness_goal, 
            user_profile.activity_level, 
            user_profile.body_type
        )
        if hasattr(daily_log, 'last_updated'):
            daily_log.last_updated = timezone.now()
            daily_log.save()
    
    # Calculate percentages safely
    def safe_percentage(value, target):
        return min(int((value / target) * 100) if target > 0 else 0, 100)
    
    protein_target = user_profile.weight * 1.6  # 1.6g protein per kg of body weight
    
    context = {
        'user_profile': user_profile,
        'daily_goals': {
            'calories_target': int(daily_calories),
            'calories_consumed': daily_log.calories_consumed,
            'calories_percentage': safe_percentage(daily_log.calories_consumed, daily_calories),
            'water_target': 3.0,
            'water_consumed': daily_log.water_consumed,
            'water_percentage': safe_percentage(daily_log.water_consumed, 3.0),
            'protein_target': int(protein_target),
            'protein_consumed': daily_log.protein_consumed,
            'protein_percentage': safe_percentage(daily_log.protein_consumed, protein_target),
            'carbs_target': int(carbs_target),
            'carbs_consumed': total_carbs,
            'carbs_percentage': safe_percentage(total_carbs, carbs_target),
            'fats_target': int(fats_target),
            'fats_consumed': total_fats,
            'fats_percentage': safe_percentage(total_fats, fats_target),
        },
        'meals': meals,
        'workout_plan': exercises,
    }
    
    return render(request, 'dashboard.html', context)

def create_default_workout(user, today, fitness_goal, activity_level, body_type):
    """Create a default workout based on the user's fitness goal, activity level, and body type"""
    # Get user profile for additional data
    try:
        user_profile = user.userprofile
        weight = user_profile.weight
        height = user_profile.height
        gender = user_profile.gender
    except UserProfile.DoesNotExist:
        weight = 70  # Default weight
        height = 170  # Default height
        gender = 'male'  # Default gender

    # Base exercises for different fitness goals - more manageable versions
    base_exercises = {
        'weight_loss': {
            'cardio': [
                {'name': 'Jumping Jacks', 'sets': 2, 'reps': 20, 'duration': 5},
                {'name': 'High Knees', 'sets': 2, 'reps': 20, 'duration': 5},
                {'name': 'Jump Rope', 'sets': 2, 'reps': 1, 'duration': 3},
                {'name': 'Jogging in Place', 'sets': 2, 'reps': 1, 'duration': 5}
            ],
            'strength': [
                {'name': 'Push-ups', 'sets': 2, 'reps': 10},
                {'name': 'Bodyweight Squats', 'sets': 2, 'reps': 15},
                {'name': 'Lunges', 'sets': 2, 'reps': 8},
                {'name': 'Plank', 'sets': 2, 'reps': 1, 'duration': 20},
                {'name': 'Russian Twists', 'sets': 2, 'reps': 15}
            ]
        },
        'muscle_gain': {
            'upper': [
                {'name': 'Push-ups', 'sets': 3, 'reps': 10},
                {'name': 'Dumbbell Rows', 'sets': 3, 'reps': 10, 'weight': 10},
                {'name': 'Dumbbell Shoulder Press', 'sets': 3, 'reps': 8, 'weight': 8},
                {'name': 'Bicep Curls', 'sets': 3, 'reps': 10, 'weight': 5}
            ],
            'lower': [
                {'name': 'Squats', 'sets': 3, 'reps': 12, 'weight': 15},
                {'name': 'Lunges', 'sets': 2, 'reps': 8, 'weight': 8},
                {'name': 'Calf Raises', 'sets': 2, 'reps': 15, 'weight': 10},
                {'name': 'Glute Bridges', 'sets': 2, 'reps': 12, 'weight': 15}
            ]
        },
        'maintenance': {
            'cardio': [
                {'name': 'Jumping Jacks', 'sets': 2, 'reps': 15, 'duration': 3},
                {'name': 'Jogging in Place', 'sets': 1, 'reps': 1, 'duration': 10},
                {'name': 'High Knees', 'sets': 2, 'reps': 15, 'duration': 3}
            ],
            'strength': [
                {'name': 'Push-ups', 'sets': 2, 'reps': 10},
                {'name': 'Bodyweight Squats', 'sets': 2, 'reps': 12},
                {'name': 'Plank', 'sets': 2, 'reps': 1, 'duration': 20},
                {'name': 'Lunges', 'sets': 2, 'reps': 8}
            ]
        },
        'endurance': {
            'cardio': [
                {'name': 'Jogging in Place', 'sets': 1, 'reps': 1, 'duration': 15},
                {'name': 'Jump Rope', 'sets': 2, 'reps': 1, 'duration': 3},
                {'name': 'High Knees', 'sets': 2, 'reps': 20, 'duration': 5}
            ],
            'strength': [
                {'name': 'Push-ups', 'sets': 2, 'reps': 15},
                {'name': 'Bodyweight Squats', 'sets': 2, 'reps': 20},
                {'name': 'Lunges', 'sets': 2, 'reps': 10},
                {'name': 'Plank', 'sets': 2, 'reps': 1, 'duration': 30}
            ]
        }
    }

    # Adjust exercises based on activity level - more moderate multipliers
    activity_multipliers = {
        'sedentary': 0.8,
        'light': 0.9,
        'moderate': 1.0,
        'active': 1.1,
        'very_active': 1.2
    }

    # Adjust exercises based on body type - more moderate adjustments
    body_type_adjustments = {
        'ectomorph': {
            'sets_multiplier': 0.9,
            'reps_multiplier': 0.9,
            'weight_multiplier': 0.9,
            'duration_multiplier': 0.9
        },
        'mesomorph': {
            'sets_multiplier': 1.0,
            'reps_multiplier': 1.0,
            'weight_multiplier': 1.0,
            'duration_multiplier': 1.0
        },
        'endomorph': {
            'sets_multiplier': 1.1,
            'reps_multiplier': 1.1,
            'weight_multiplier': 1.1,
            'duration_multiplier': 1.1
        }
    }

    # Adjust exercises based on weight - more moderate adjustments
    weight_multiplier = min(max(weight / 70, 0.9), 1.1)  # Normalize around 70kg with smaller range

    # Get base exercises for the fitness goal
    goal_exercises = base_exercises.get(fitness_goal, base_exercises['maintenance'])
    
    # Apply activity level and body type adjustments
    activity_multiplier = activity_multipliers.get(activity_level, 1.0)
    body_type_adjustment = body_type_adjustments.get(body_type, body_type_adjustments['mesomorph'])
    
    # Log the adjustments being applied
    logger.info(f"Creating workout for user {user.username} with:")
    logger.info(f"Fitness goal: {fitness_goal}")
    logger.info(f"Activity level: {activity_level} (multiplier: {activity_multiplier})")
    logger.info(f"Body type: {body_type} (adjustments: {body_type_adjustment})")
    logger.info(f"Weight: {weight}kg (multiplier: {weight_multiplier})")
    
    # Delete any existing exercises for today
    Exercise.objects.filter(user=user, date=today).delete()
    
    # Combine exercises from different categories
    all_exercises = []
    for category in goal_exercises.values():
        for exercise in category:
            adjusted_exercise = exercise.copy()
            
            # Adjust sets
            if 'sets' in adjusted_exercise:
                adjusted_exercise['sets'] = int(adjusted_exercise['sets'] * 
                                             activity_multiplier * 
                                             body_type_adjustment['sets_multiplier'] *
                                             weight_multiplier)
            
            # Adjust reps
            if 'reps' in adjusted_exercise:
                adjusted_exercise['reps'] = int(adjusted_exercise['reps'] * 
                                             activity_multiplier * 
                                             body_type_adjustment['reps_multiplier'] *
                                             weight_multiplier)
            
            # Adjust weight
            if 'weight' in adjusted_exercise:
                adjusted_exercise['weight'] = int(adjusted_exercise['weight'] * 
                                               activity_multiplier * 
                                               body_type_adjustment['weight_multiplier'] *
                                               weight_multiplier)
            
            # Adjust duration
            if 'duration' in adjusted_exercise:
                adjusted_exercise['duration'] = int(adjusted_exercise['duration'] * 
                                                 activity_multiplier * 
                                                 body_type_adjustment['duration_multiplier'] *
                                                 weight_multiplier)
            
            # Ensure minimum values
            adjusted_exercise['sets'] = max(1, adjusted_exercise.get('sets', 1))
            adjusted_exercise['reps'] = max(1, adjusted_exercise.get('reps', 1))
            if 'weight' in adjusted_exercise:
                adjusted_exercise['weight'] = max(1, adjusted_exercise['weight'])
            if 'duration' in adjusted_exercise:
                adjusted_exercise['duration'] = max(1, adjusted_exercise['duration'])
            
            all_exercises.append(adjusted_exercise)
    
    # Log the final exercises
    logger.info(f"Created {len(all_exercises)} exercises:")
    for exercise in all_exercises:
        logger.info(f"- {exercise['name']}: {exercise}")
    
    # Create the exercises
    for exercise_data in all_exercises:
        Exercise.objects.create(
            user=user,
            date=today,
            **exercise_data
        )
    
    # Return the newly created exercises
    return Exercise.objects.filter(user=user, date=today)

@login_required
def add_meal(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            meal_type = request.POST.get('meal_type')
            calories = int(request.POST.get('calories'))
            protein = float(request.POST.get('protein'))
            carbs = float(request.POST.get('carbs'))
            fats = float(request.POST.get('fats'))
            time_str = request.POST.get('time')
            tags_json = request.POST.get('tags', '[]')
            
            # Convert time string to time object
            time_obj = datetime.strptime(time_str, '%H:%M').time()
            
            # Parse tags from JSON
            try:
                tags = json.loads(tags_json)
            except json.JSONDecodeError:
                tags = []
            
            # Create meal
            meal = Meal.objects.create(
                user=request.user,
                name=name,
                meal_type=meal_type,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fats=fats,
                date=date.today(),
                time=time_obj,
                tags=tags
            )
            
            # Update daily log
            daily_log, created = DailyLog.objects.get_or_create(
                user=request.user,
                date=date.today()
            )
            daily_log.calories_consumed += calories
            daily_log.protein_consumed += protein
            daily_log.save()
            
            return JsonResponse({'status': 'success', 'message': 'Meal added successfully'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required
def complete_exercise(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            exercise_id = data.get('exercise_id')
            
            exercise = Exercise.objects.get(id=exercise_id, user=request.user)
            exercise.completed = True
            exercise.save()
            
            return JsonResponse({'status': 'success'})
        except Exercise.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Exercise not found'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required
def profile(request):
    # Get or create UserProfile
    user_profile, created = UserProfile.objects.get_or_create(
        user=request.user,
        defaults={
            'height': 170,
            'weight': 70,
            'body_type': 'mesomorph',
            'activity_level': 'moderate',
            'fitness_goal': 'maintenance',
            'dietary_preferences': [],
            'health_conditions': ''
        }
    )
    
    if request.method == 'POST':
        # Update user profile
        user_profile.height = float(request.POST.get('height'))
        user_profile.weight = float(request.POST.get('weight'))
        user_profile.body_type = request.POST.get('body_type')
        user_profile.activity_level = request.POST.get('activity_level')
        user_profile.fitness_goal = request.POST.get('fitness_goal')
        user_profile.dietary_preferences = request.POST.getlist('dietary_preferences')
        user_profile.health_conditions = request.POST.get('health_conditions')
        user_profile.save()
        
        return redirect('dashboard')
    
    return render(request, 'profile.html', {'user_profile': user_profile})

@login_required
def chat(request):
    try:
        profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(
            user=request.user,
            height=170,
            weight=70,
            body_type='mesomorph',
            activity_level='moderate',
            fitness_goal='maintenance'
        )
        return redirect('profile')

    if request.method == 'POST':
        message = request.POST.get('message')
        if message:
            try:
                # Create chat history with Groq API
                client = OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_API_BASE
                )
                
                # Prepare the system message with user's health data
                system_message = f"""You are a helpful AI assistant for a health and fitness tracking app. 
                The user's profile:
                - Height: {profile.height}cm
                - Weight: {profile.weight}kg
                - Body Type: {profile.body_type}
                - Activity Level: {profile.activity_level}
                - Fitness Goal: {profile.fitness_goal}
                - BMI: {profile.bmi:.1f}
                - BMI Category: {profile.bmi_category}
                - BMR: {profile.bmr:.0f} calories
                - Daily Calorie Needs: {profile.daily_calorie_needs:.0f} calories
                
                Provide personalized advice based on their profile. Structure your responses in the following format:

                **1. Summary**
                - Brief overview of the main points
                - Key takeaway

                **2. Detailed Analysis**
                - Break down the topic into clear categories
                - Use bullet points for each category
                - Include specific numbers and metrics where relevant

                **3. Action Items**
                - List 3-5 specific, actionable steps
                - Prioritize the most important actions
                - Include timeframes where applicable

                **4. Additional Resources**
                - Suggest relevant tools or resources
                - Include links to helpful content
                - Mention any apps or trackers that could help

                **5. Next Steps**
                - What to do immediately
                - What to monitor
                - When to check back

                Use markdown formatting to structure your response:
                - Use **bold** for section headers
                - Use *italics* for emphasis
                - Use bullet points (-) for lists
                - Use numbered lists (1., 2., 3.) for steps
                - Keep paragraphs short and focused
                - Be encouraging and positive"""
                
                response = client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                ai_response = response.choices[0].message.content
                
                # Save the chat history
                ChatHistory.objects.create(
                    user=request.user,
                    message=message,
                    response=ai_response
                )
                
                return JsonResponse({
                    'status': 'success',
                    'response': ai_response
                })
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'message': str(e)
                })
    
    # Get recent chat history
    chat_history = ChatHistory.objects.filter(user=request.user).order_by('-timestamp')[:10]
    return render(request, 'chat.html', {
        'chat_history': chat_history,
        'profile': profile
    })

@login_required
def update_water(request):
    if request.method == 'POST':
        try:
            # Log the raw request body for debugging
            logger.info(f"Received water update request: {request.body}")
            
            data = json.loads(request.body)
            water_amount = float(data.get('water_amount', 0))
            
            if water_amount <= 0:
                logger.warning(f"Invalid water amount: {water_amount}")
                return JsonResponse({'status': 'error', 'message': 'Invalid water amount'})
            
            today = date.today()
            
            try:
                user_profile = request.user.userprofile
            except UserProfile.DoesNotExist:
                logger.error(f"User profile not found for user {request.user.username}")
                return JsonResponse({'status': 'error', 'message': 'User profile not found'})
            
            # Get or create daily log
            daily_log, created = DailyLog.objects.get_or_create(
                user=request.user,
                date=today,
                defaults={'weight': user_profile.weight, 'water_consumed': 0}
            )
            
            # Check if the log has been processed in the current session
            log_session_key = f"daily_log_{daily_log.id}_processed"
            if created or daily_log.date != today or not request.session.get(log_session_key, False):
                # New log, new day, or first time in this session - reset water
                daily_log.water_consumed = 0
                request.session[log_session_key] = True
                logger.info(f"Reset water consumption for user {request.user.username} in session {SERVER_SESSION_ID}")
            
            # Update water consumed
            daily_log.water_consumed = float(daily_log.water_consumed or 0) + water_amount
            daily_log.save()
            
            logger.info(f"Successfully updated water intake for user {request.user.username}: {water_amount}L, total: {daily_log.water_consumed}L")
            return JsonResponse({
                'status': 'success', 
                'message': f'Added {water_amount}L of water',
                'new_total': daily_log.water_consumed
            })
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': 'Invalid data format'})
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': 'Invalid water amount'})
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JsonResponse({'status': 'error', 'message': f'An error occurred: {str(e)}'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def register(request):
    """Register a new user"""
    if request.method == 'POST':
        # Get form data
        username = request.POST.get('username')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        # Validation
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return render(request, 'auth/register.html')
        
        if password != password_confirm:
            messages.error(request, 'Passwords do not match')
            return render(request, 'auth/register.html')
        
        if len(password) < 8:
            messages.error(request, 'Password must be at least 8 characters')
            return render(request, 'auth/register.html')
        
        # Create user
        user = User.objects.create_user(
            username=username,
            password=password
        )
        
        # Login user automatically
        login(request, user)
        
        # UserProfile will be created by the signal in models.py
        
        # Redirect to profile page to complete setup
        return redirect('profile')
    
    return render(request, 'auth/register.html') 