from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    BODY_TYPE_CHOICES = [
        ('ectomorph', 'Ectomorph'),
        ('mesomorph', 'Mesomorph'),
        ('endomorph', 'Endomorph'),
    ]
    
    ACTIVITY_LEVEL_CHOICES = [
        ('sedentary', 'Sedentary'),
        ('light', 'Light'),
        ('moderate', 'Moderate'),
        ('active', 'Active'),
        ('very_active', 'Very Active'),
    ]
    
    FITNESS_GOAL_CHOICES = [
        ('weight_loss', 'Weight Loss'),
        ('muscle_gain', 'Muscle Gain'),
        ('maintenance', 'Maintenance'),
        ('endurance', 'Endurance'),
    ]
    
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default='male')
    height = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(300)],
        help_text="Height in centimeters"
    )
    weight = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(500)],
        help_text="Weight in kilograms"
    )
    body_type = models.CharField(max_length=20, choices=BODY_TYPE_CHOICES)
    activity_level = models.CharField(max_length=20, choices=ACTIVITY_LEVEL_CHOICES)
    fitness_goal = models.CharField(max_length=20, choices=FITNESS_GOAL_CHOICES)
    dietary_preferences = models.JSONField(default=list)
    health_conditions = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def bmi(self):
        height_in_meters = self.height / 100
        return round(self.weight / (height_in_meters ** 2), 2)
    
    @property
    def bmi_category(self):
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    @property
    def bmr(self):
        """Calculate Basal Metabolic Rate using Harris-Benedict equation"""
        # Basic BMR calculation
        bmr = (10 * self.weight) + (6.25 * self.height) - (5 * 25)
        # Adjust for gender
        bmr += 5 if self.gender == 'male' else -161
        return bmr
    
    @property
    def daily_calorie_needs(self):
        """Calculate daily calorie needs based on activity level and fitness goal"""
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        
        # Calculate BMR and adjust for body type
        calorie_multiplier = 1.2 if self.body_type == 'ectomorph' else (
            1.1 if self.body_type == 'mesomorph' else 1.0
        )
        
        # Calculate daily calories
        daily_calories = self.bmr * activity_multipliers[self.activity_level] * calorie_multiplier
        
        # Adjust based on fitness goal
        if self.fitness_goal == 'weight_loss':
            daily_calories *= 0.85
        elif self.fitness_goal == 'muscle_gain':
            daily_calories *= 1.15
            
        return daily_calories
    
    def __str__(self):
        return f"{self.user.username}'s Profile"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create a UserProfile for every new User."""
    if created:
        UserProfile.objects.create(
            user=instance,
            height=170,  # Default height
            weight=70,   # Default weight
            body_type='mesomorph',
            activity_level='moderate',
            fitness_goal='maintenance'
        )

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save the UserProfile whenever its User is saved."""
    try:
        instance.userprofile.save()
    except UserProfile.DoesNotExist:
        UserProfile.objects.create(
            user=instance,
            height=170,
            weight=70,
            body_type='mesomorph',
            activity_level='moderate',
            fitness_goal='maintenance'
        )

class DailyLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    weight = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(500)])
    calories_consumed = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    water_consumed = models.FloatField(default=0, validators=[MinValueValidator(0)])
    protein_consumed = models.FloatField(default=0, validators=[MinValueValidator(0)])
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user', 'date']
        ordering = ['-date']

    def __str__(self):
        return f"{self.user.username}'s Log - {self.date}"

class Meal(models.Model):
    MEAL_TYPE_CHOICES = [
        ('breakfast', 'Breakfast'),
        ('lunch', 'Lunch'),
        ('dinner', 'Dinner'),
        ('snack', 'Snack'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    meal_type = models.CharField(max_length=20, choices=MEAL_TYPE_CHOICES)
    calories = models.IntegerField(validators=[MinValueValidator(0)])
    protein = models.FloatField(validators=[MinValueValidator(0)])
    carbs = models.FloatField(validators=[MinValueValidator(0)])
    fats = models.FloatField(validators=[MinValueValidator(0)])
    date = models.DateField()
    time = models.TimeField()
    tags = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['date', 'time']

    def __str__(self):
        return f"{self.name} ({self.meal_type}) - {self.date}"

class Exercise(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    sets = models.IntegerField(validators=[MinValueValidator(0)])
    reps = models.IntegerField(validators=[MinValueValidator(0)])
    weight = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0)])
    duration = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Duration in minutes"
    )
    date = models.DateField()
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['date', 'created_at']

    def __str__(self):
        return f"{self.name} - {self.date}"

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Chat with {self.user.username} at {self.created_at}"

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Chat histories'

    def __str__(self):
        return f"Chat with {self.user.username} at {self.timestamp}"

class ServerSession(models.Model):
    """A model to track server sessions for resetting daily values on server restart"""
    session_id = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Server session {self.session_id} started at {self.created_at}"
