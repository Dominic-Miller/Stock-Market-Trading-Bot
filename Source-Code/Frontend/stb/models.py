from django.db import models

# Create your models here.
from django.db import models

class Trade(models.Model):
    # Define fields for your Trade model
    symbol = models.CharField(max_length=50)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.PositiveIntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.symbol} - {self.timestamp}"
