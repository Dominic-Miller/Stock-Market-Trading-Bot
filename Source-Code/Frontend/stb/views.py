from django.shortcuts import render, redirect, get_object_or_404
from .models import Trade  
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


def index(request):
    trades = Trade.objects.all()  # Retrieve trading data from the database
    return render(request, 'index.html', {'trades': trades})

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

@login_required
def profile(request, username):
    user = get_object_or_404(User, username=username)
    return render(request, 'profile.html', {'user': user})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to a success page.
            return render(request, 'profile.html')
        else:
            # Return an 'invalid login' error message.
            return render(request, 'login.html', {'error': 'Invalid login credentials'})
    else:
        return render(request, 'login.html')
    
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return render(request, 'dashboard.html')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def about(request):
    return render(request, 'about.html')
