from django import forms

MODEL_CHOICES = [
    ('lstm', 'LSTM Neural Network'),
    # ('svm', 'Support Vector Machine (SVM)'),
    ('random_forest', 'Random Forest'),
    # ('knn', 'K-Nearest Neighbors (KNN)'),
    # ('decision_tree', 'Decision Tree'),
]

class PredictionForm(forms.Form):
    city = forms.CharField(
        max_length=100,
        label='City Name (in India)',
        initial='Mumbai',
        widget=forms.TextInput(attrs={
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'placeholder': 'e.g. Mumbai, Pune, Delhi, Bengaluru',
            'aria-label': 'City name'
        }),
        help_text='Enter the city name for AQI prediction'
    )

    forecast_type = forms.ChoiceField(
        choices=[('single', 'Single Date/Time'), ('range', 'Date Range')],
        initial='single',
        widget=forms.RadioSelect(attrs={
            'class': 'flex gap-4 mb-4'
        }),
        label='Forecast Type'
    )

    prediction_datetime = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={
            'type': 'datetime-local',
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'aria-label': 'Select prediction date and time'
        }),
        label='Select Date and Time for Prediction'
    )

    start_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'aria-label': 'Select start date'
        }),
        label='Select Start Date for Range'
    )

    end_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'aria-label': 'Select end date'
        }),
        label='Select End Date for Range'
    )
    
    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Select Prediction Model',
        initial='random_forest',  
        widget=forms.Select(attrs={
            'class': 'border rounded px-3 py-2 w-full',
            'aria-label': 'Select machine learning model'
        }),
        help_text='Choose the machine learning model for AQI prediction'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model'].widget.attrs.update({
            'onchange': 'this.form.classList.add("changed")'  # Optional: adds visual feedback on model change
        })


class RangePredictionForm(forms.Form):
    city = forms.CharField(
        max_length=100,
        label='City Name (in India)',
        initial='Mumbai',
        widget=forms.TextInput(attrs={
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'placeholder': 'e.g. Mumbai, Pune, Delhi',
            'aria-label': 'City name'
        }),
        help_text='Enter the city name for AQI range prediction'
    )

    start_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'aria-label': 'Select start date'
        }),
        label='Start Date for Prediction'
    )

    end_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'border rounded px-3 py-2 w-full mb-4',
            'aria-label': 'Select end date'
        }),
        label='End Date for Prediction'
    )

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Select Prediction Model',
        initial='random_forest',
        widget=forms.Select(attrs={
            'class': 'border rounded px-3 py-2 w-full',
            'aria-label': 'Select machine learning model'
        }),
        help_text='Choose the machine learning model for range prediction'
    )