# Importing modules
from astropy.io import fits
import matplotlib.pylab as plt
imprt matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.cosmology import Planck13
cosmo=Planck13
from astropy . coordinates import Distance
from astropy import units as u
from astropy.table import Table
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm
import random
import pyarrow.parquet as pq
import flexcode
from flexcode.regression_models import XGBoost
from flexcode import post_processing
from sklearn import preprocessing

plt.rcParams.update({
    "font.size": 8,  # Base font size
    "figure.figsize": (3.3, 2.5),
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.usetex": False,
    "font.family": "serif",
    "figure.dpi": 300,  # For raster export
    "savefig.dpi": 600  # Higher DPI for saved files
})

# List of columns you want to read
columns_to_read = ['id_galaxy_sky', 'id_group_sky', 'ra', 'dec', 'zobs', 'zcos', 'total_ap_dust_Z_VISTA', 'total_ab_dust_Z_VISTA', 'total_ap_dust_Y_VISTA', 'total_ap_dust_J_VISTA', 'total_ap_dust_H_VISTA', 'total_ap_dust_K_VISTA','total_ap_dust_u_VST','total_ab_dust_u_VST',  'total_ap_dust_g_VST', 'total_ap_dust_r_VST', 'total_ab_dust_r_VST', 'total_ap_dust_i_VST']

# Read only those columns
table = pq.read_table('/mnt/lustre/projects/astro/general/sp624/waves_mocks/v0.3.0/wide/waves_wide_gals.parquet', columns=columns_to_read)

# Optional: convert to pandas if needed
df = table.to_pandas()

# Importing Euclid FS2 to DataFrame, and renaming some columns
#hdul = fits.open('../Photo_group_project/11401.fits')
#data = hdul[1].data
#cols = hdul[1].columns
#t=Table(data)
#df = t.to_pandas()
df = df.rename(columns={"ra": "RA"})
df = df.rename(columns={"dec": "Dec"})
df = df.rename(columns={"zobs": "Z"})
#df = df.rename(columns={"total_ap_dust_Z_VISTA": "Rpetro"})
df = df.rename(columns={'id_group_sky': "HaloID"})
del table

# Magnitude limiting to r<21
df=df[(df['total_ap_dust_Z_VISTA']<21)]

#only waves-N 
df=df[(df['Dec']>-10)]

regions = [
    {"ra_min": 193.0, "ra_max": 205.0, "dec_min": -2.0, "dec_max": 3.0},   # G09 look alike
    {"ra_min": 174.0, "ra_max": 186.0, "dec_min": -3.0, "dec_max": 2.0},   # G12
    {"ra_min": 211.5, "ra_max": 223.5, "dec_min": -2.0, "dec_max": 3.0},   # G15
    {"ra_min": 165, "ra_max": 170, "dec_min": -2.0, "dec_max": 3.0},   #25 sq deg pencil beam
]

# Apply the filters
mask = (
    ((df["RA"] >= regions[0]["ra_min"]) & (df["RA"] <= regions[0]["ra_max"]) &
     (df["Dec"] >= regions[0]["dec_min"]) & (df["Dec"] <= regions[0]["dec_max"]))
    |
    ((df["RA"] >= regions[1]["ra_min"]) & (df["RA"] <= regions[1]["ra_max"]) &
     (df["Dec"] >= regions[1]["dec_min"]) & (df["Dec"] <= regions[1]["dec_max"]))
    |
    ((df["RA"] >= regions[2]["ra_min"]) & (df["RA"] <= regions[2]["ra_max"]) &
     (df["Dec"] >= regions[2]["dec_min"]) & (df["Dec"] <= regions[2]["dec_max"]))
        |
    ((df["RA"] >= regions[3]["ra_min"]) & (df["RA"] <= regions[3]["ra_max"]) &
     (df["Dec"] >= regions[3]["dec_min"]) & (df["Dec"] <= regions[3]["dec_max"]))
)

# Filter the dataframe
df = df[mask]

waves_mag_cols = ['total_ap_dust_Z_VISTA', 'total_ap_dust_Y_VISTA', 'total_ap_dust_J_VISTA', 'total_ap_dust_H_VISTA', 'total_ap_dust_K_VISTA','total_ap_dust_u_VST', 'total_ap_dust_g_VST', 'total_ap_dust_r_VST', 'total_ap_dust_i_VST']

# Conversion of flux to magnitude
def flux_to_mag(flux):
    return -2.5 * np.log10(flux) - 48.6

def mag_to_flux(mag):
    return 10 ** (-(mag + 48.6) / 2.5)


flux_columns = []
for i in waves_mag_cols:
    flux_columns.append(i+'_flux')
    df[i+'_flux'] = mag_to_flux(df[i])

# Applying a 1% error to all fluxes
def apply_random_error(flux):
    error_factor = 0.05  
    noise = np.random.normal(loc=0, scale=error_factor * flux)
    return flux + noise


for col in tqdm(flux_columns):
    df[col] = df[col].apply(apply_random_error)


# Converting flux to magnitudes
mag_cols=[]

for col in tqdm(flux_columns):
    mag_col = f'mag_{col}'  # New column name for magnitudes
    mag_cols.append(mag_col)
    df[mag_col] = df[col].apply(flux_to_mag)

# Creating X, the array of features needed for photo-z estimate

X=np.zeros((len(df),45))

l=0
for i in tqdm(range(len(mag_cols))):
    for j in range(i+1,len(mag_cols)):
        X[:,l]=(df[mag_cols[i]] - df[mag_cols[j]])
        l+=1
for mag in tqdm(mag_cols):
    X[:,l]=df[mag]
    l+=1


# Making of column of 0-N in the dataframe to be used later
df['raw_index']=range(len(df))


def print_section(title):
    """Helper function to print formatted section headers"""
    print(f"\n{'=' * 50}")
    print(f"    {title}")
    print(f"{'=' * 50}")


print_section("DATA PREPARATION")

# Split the data into training and testing sets based on Right Ascension (RA)
# Note: df['raw_index'] was previously created with: df['raw_index'] = range(len(df))
print("Creating training and test sets based on RA coordinates...")
dftrain = df[(df['RA'] < 170)]
dftest = df[(df['RA'] > 170)]

# Report dataset sizes
print(f"Training set size: {len(dftrain)} galaxies")
print(f"Test set size: {len(dftest)} galaxies")

# Extract feature matrices using the raw_index created earlier
print("Extracting feature matrices and spectroscopic redshifts...")
Xtrain = X[dftrain['raw_index']]
Xtest = X[dftest['raw_index']]  # Using proper test indices

# Scale features to have zero mean and unit variance
print("Scaling features...")
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

# Extract spectroscopic redshift values (ground truth)
Ztrain = dftrain['Z'].values
Ztest = dftest['Z'].values

print(f"Feature matrix shapes - Training: {Xtrain.shape}, Testing: {Xtest.shape}")
print(f"Redshift vector shapes - Training: {Ztrain.shape}, Testing: {Ztest.shape}")

print_section("MODEL TRAINING")

# Split training data into actual training and validation sets (80/20 split)
n_obs = Xtrain.shape[0]
n_train = round(n_obs * 0.8)
n_validation = n_obs - n_train

# Randomly shuffle the training data
print(f"Splitting training data: {n_train} for training, {n_validation} for validation")
perm = np.random.permutation(n_obs)
Xtrain_final = Xtrain[perm[:n_train], :]
Ztrain_final = Ztrain[perm[:n_train]]
X_validation = Xtrain[perm[n_train:], :]
Z_validation = Ztrain[perm[n_train:]]

# Set up and train the FlexCode model with XGBoost as the regression method
print("Training FlexCode model with XGBoost regression...")
model = flexcode.FlexCodeModel(
    XGBoost, 
    max_basis=10, 
    basis_system='cosine',
    regression_params={"max_depth": 8}
)

# Fit the model on training data
model.fit(Xtrain_final, Ztrain_final)
print("Model fitted. Now tuning hyperparameters using validation data...")
model.tune(X_validation, Z_validation)
print("Model training complete!")

print_section("PREDICTION")

# Generate conditional density estimates for each test galaxy
print("Predicting conditional density estimates for test galaxies...")
cdes, z_grid = model.predict(Xtest, n_grid=1000)
print(f"Generated {cdes.shape[0]} conditional density estimates across {cdes.shape[1]} redshift grid points")

# Keep copies of the original CDEs for comparison and post-processing
print("Creating copies of density estimates for different post-processing methods...")
cdes_sharp = cdes.copy()

print_section("POST-PROCESSING OPTIMIZATION")

# Function to create appropriate samples for optimization
def create_optimization_samples(cdes_array, ztest_array, target_size=10000):
    """Create appropriate-sized samples for optimization procedures"""
    if len(cdes_array) > target_size:
        # If dataset is large, sample a subset
        sample_indices = np.random.choice(len(cdes_array), size=target_size, replace=False)
        cdes_sample = cdes_array[sample_indices]
        ztest_sample = ztest_array[sample_indices]
        print(f"Using {target_size} random samples for optimization")
    else:
        # If dataset is small, use all data
        cdes_sample = cdes_array
        ztest_sample = ztest_array
        print(f"Using all {len(cdes_array)} samples for optimization")
    return cdes_sample, ztest_sample

# Sample data for bump threshold optimization
print("Preparing for bump removal threshold optimization...")
cdes_sample, ztest_sample = create_optimization_samples(cdes, Ztest)

# Optimize the bump removal threshold
print("Optimizing the bump removal threshold...")
delta_grid = np.logspace(-1, 1, 50)  # Grid of potential thresholds
best_delta = post_processing.choose_bump_threshold(cdes_sample, z_grid, ztest_sample, delta_grid)
print(f"Optimal bump removal threshold: {best_delta}")

# Sample data for sharpening parameter optimization
print("Preparing for sharpening parameter optimization...")
cdes_sample, ztest_sample = create_optimization_samples(cdes, Ztest)

# Optimize the sharpening parameter
print("Optimizing the sharpening parameter...")
alpha_grid = np.logspace(-1, 1, 50)  # Grid of potential sharpening values
best_alpha = post_processing.choose_sharpen(cdes_sample, z_grid, ztest_sample, alpha_grid)
print(f"Optimal sharpening parameter: {best_alpha}")

print_section("APPLYING POST-PROCESSING")



# Apply bump removal to each density estimate
print("Applying bump removal to each density estimate...")
for i in tqdm(range(len(cdes_sharp))):
    post_processing._remove_bumps(cdes_sharp[i], best_delta)  # Using optimized delta
    
# Apply sharpening to each density estimate
print("Applying sharpening to each density estimate...")
for i in tqdm(range(len(cdes_sharp))):
    post_processing.sharpen(cdes_sharp[i], best_alpha)  # Using optimized alpha

# Normalize the density estimates after post-processing
print("Normalizing density estimates...")
for i in range(len(cdes_sharp)):
    post_processing.normalize(cdes_sharp[i])

print_section("COMPLETED")
print("Photometric redshift estimation pipeline completed successfully!")

np.save('cdes_sharp',cdes_sharp)
np.save('cdes',cdes)
np.save('z_grid',z_grid)




def plot_example_redshifts():
    print("Plotting example photometric redshift distributions...")
    
    # Create figure and grid
    fig = plt.figure(figsize=(20, 12))
    spec = gridspec.GridSpec(ncols=3, nrows=2)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Create subplots
    axes = [
        fig.add_subplot(spec[0, 0]),
        fig.add_subplot(spec[0, 1]),
        fig.add_subplot(spec[1, 0]),
        fig.add_subplot(spec[1, 1]),
        fig.add_subplot(spec[1, 2])
    ]
    
    # Get random sample of test galaxies
    # Ensure we're only selecting from our test indices
    test_indices = dftest['raw_index'].values
    sample_indices = random.sample(range(len(test_indices)), 5)
    sample_galaxies = [test_indices[i] for i in sample_indices]
    
    # Plot each sample
    for i, (ax, galaxy_idx) in enumerate(zip(axes, sample_galaxies)):
        # Find the position in our test array that corresponds to this galaxy
        cde_idx = np.where(dftest['raw_index'] == galaxy_idx)[0][0]
        
        # Plot the pre and post processed distributions
        ax.plot(z_grid, cdes[cde_idx], label='Pre-processing' if i == 0 else None)
        ax.plot(z_grid, cdes_sharp[cde_idx], label='Post-processing' if i == 0 else None)
        
        # Plot the true redshift
        true_z = dftest['Z'].iloc[cde_idx]
        ax.axvline(true_z, color="k", label='True $z$' if i == 0 else None, 
                  linewidth=2, linestyle='dashed')
        
        # Format the axes
        ax.set_yticks([])
        #ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xlabel('$z$')
        ax.set_title(f'Galaxy ID = {galaxy_idx}', fontsize=18)
    
    # Add legend to the figure
    fig.legend(loc=(0.7, 0.8), frameon=False)
    
    # Save and show the figure
    plt.savefig('photometric_redshifts.jpg', bbox_inches='tight', dpi=200)
    plt.clf()
    
    print("Plot saved to 'photometric_redshifts.jpg'")

# Call the plotting function
plot_example_redshifts()

def calculate_point_estimates():
    print("Calculating point estimates from photometric redshift distributions...")
    
    # Important: We need to work with the test set indices
    # Our cdes_sharp array corresponds to the dftest dataframe in the same order
    
    # Initialize arrays to store the results
    n_galaxies = len(dftest)
    zpredpeak = np.zeros(n_galaxies)
    zprednorm = np.zeros(n_galaxies)
    
    # Flatten z_grid for easier use
    z_grid_flat = z_grid[:, 0]
    
    # Vectorized calculation for the mean (expectation) of each distribution
    print("Calculating mean redshift for each galaxy...")
    # We can vectorize this calculation across all galaxies
    # This computes the weighted average (mean) of z values for each CDE
    z_grid_2d = z_grid_flat.reshape(1, -1)  # Shape (1, n_grid)
    weights_sum = np.sum(cdes_sharp, axis=1, keepdims=True)  # Shape (n_galaxies, 1)
    zprednorm = np.sum(z_grid_2d * cdes_sharp, axis=1) / weights_sum.flatten()
    
    # For the mode, we still need a loop since argmax is applied per distribution
    print("Calculating peak (mode) redshift for each galaxy...")
    for i in tqdm(range(n_galaxies)):
        # Find the index of the maximum value
        max_idx = np.argmax(cdes_sharp[i])
        # Get the corresponding redshift value
        zpredpeak[i] = z_grid_flat[max_idx]
    
    # Add the results to the test dataframe
    dftest['z_peak'] = zpredpeak
    dftest['z_mean'] = zprednorm
    
    # Calculate statistics about the estimates
    print("\nPoint estimate statistics:")
    print(f"Mean of peak estimates: {zpredpeak.mean():.4f}")
    print(f"Mean of expectation estimates: {zprednorm.mean():.4f}")
    print(f"Median of peak estimates: {np.median(zpredpeak):.4f}")
    print(f"Median of expectation estimates: {np.median(zprednorm):.4f}")
    
    # Calculate errors compared to spectroscopic redshifts
    z_spec = dftest['Z'].values
    peak_err = zpredpeak - z_spec
    mean_err = zprednorm - z_spec
    
    print(f"\nMean absolute error (peak): {np.abs(peak_err).mean():.4f}")
    print(f"Mean absolute error (mean): {np.abs(mean_err).mean():.4f}")
    print(f"Root mean square error (peak): {np.sqrt(np.mean(peak_err**2)):.4f}")
    print(f"Root mean square error (mean): {np.sqrt(np.mean(mean_err**2)):.4f}")
    
    return zpredpeak, zprednorm

# Call the function to calculate point estimates
zpredpeak, zprednorm = calculate_point_estimates()

dftest['Zpeak']=zpredpeak
dftest['Znorm']=zprednorm


# Plotting the predicted z against the spec-z

sample=dftest[(dftest['Zpeak']>0.004) & (dftest['Zpeak']<0.5)]
plt.scatter(sample['Z'],
            sample['Zpeak'],
            s=1,c='k',label='Galaxies',linewidth=0)
im=plt.hexbin(sample['Z'],sample['Zpeak'],gridsize=(100,round(100*2/3)),
              cmap='turbo',extent=(-0.01,0.5,-0.01,0.5),mincnt=10,linewidths=0,norm=matplotlib.colors.LogNorm())

plt.plot([0,0.5],[0,0.5],c='k',linestyle='dashed',linewidth=2)
plt.xlabel(r'$z_\mathrm{spec}$')#,fontsize=20)
plt.ylabel(r'$z_\mathrm{photo}$')#,fontsize=20)
plt.xlim(-0.01,0.5)
plt.ylim(-0.01,0.5)

cbar=plt.colorbar().set_label(label='Counts')
#im.figure.axes[1].tick_params(labelsize=20)

plt.savefig('z_photo_z_spec.jpg',bbox_inches='tight',dpi=200)

