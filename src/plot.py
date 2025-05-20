import matplotlib.pyplot as plt
import numpy as np

# Fire risk function based on realistic NYC trends
def fire_risk(age):
    if age <= 20:
        return 5 + 0.2 * age            # newer buildings: very low risk
    elif age <= 50:
        return 9 + 0.5 * (age - 20)     # moderate aging: rising risk
    elif age <= 100:
        return 24 + 0.3 * (age - 50)    # old systems, poor fireproofing
    else:
        return 39 - 0.1 * (age - 100)   # renovation bias lowers risk slightly

# Generate building ages 0 to 150
ages = np.arange(0, 151)
risks = [fire_risk(age) for age in ages]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ages, risks, color='firebrick', linewidth=2, label='Estimated Fire Risk')

# Annotate important years
plt.axvline(55, color='gray', linestyle='--', label='1968 NYC Building Code')
plt.axvline(15, color='gray', linestyle=':', label='2008 Code Overhaul')
plt.annotate("1968 Building Code", xy=(55, fire_risk(55)+2), xytext=(60, 45),
             arrowprops=dict(arrowstyle="->", color='gray'), fontsize=9)
plt.annotate("2008 Code", xy=(15, fire_risk(15)+2), xytext=(20, 30),
             arrowprops=dict(arrowstyle="->", color='gray'), fontsize=9)

# Labels and legend
plt.title("Estimated Fire Risk vs Building Age in NYC", fontsize=14)
plt.xlabel("Building Age (years)", fontsize=12)
plt.ylabel("Estimated Fire Risk (%)", fontsize=12)
plt.ylim(0, 50)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
