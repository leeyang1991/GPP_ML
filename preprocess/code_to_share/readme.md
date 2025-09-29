The script automates the workflow of:

1. Generating rectangular polygons around flux observation sites.
2. Downloading, preprocessing, and organizing **NASA HLS (Harmonized Landsat and Sentinel-2) imagery** from **Google Earth Engine (GEE)** for those sites.

------

### **Key Components**

#### **1. Expand_points_to_rectangle class**

- **Input:** Flux site metadata (site IDs, latitudes, longitudes).
- **Process:**
  - Reads flux site coordinates from pre-saved dataframes (`fluxdata_DD.df`, `metadata.df`).
  - Expands each site into a rectangle (~25 pixels × 30 m ≈ 750 m buffer).
  - Builds bounding box polygons (shapely `Polygon`).
- **Output:** A shapefile (`sites.shp`) containing rectangles for all sites.

------

#### **2. Download_from_GEE class**

Handles downloading and preprocessing HLS data.

- **Initialization:**
  - Selects the Sentinel-based HLS collection: `NASA/HLS/HLSS30/v002`.
  - Authenticates and initializes Earth Engine (`ee.Initialize`).
- **Main Steps in `run()`:**
  1. **Download images per year (2016–2024):**
     - Uses rectangles from `sites.shp`.
     - Queries GEE for imagery intersecting each site.
     - Exports B2–B7 spectral bands + Fmask to `.zip` files (parallelized).
  2. **Check:** Validates downloaded zips.
  3. **Unzip:** Extracts imagery into organized folders.
  4. **Resize to 50×50 pixels:**
     - Clips each image array to 50×50.
     - Ensures no invalid raster dimensions.
  5. **Quality Control:**
     - Filters out cloudy, shadowed, water, and snow/ice pixels using Fmask values.
  6. **Fill NaN values (optional):** Replace NaN with zeros.
  7. **Merge Bands:** Combines valid spectral bands into a multi-band GeoTIFF, discarding images with too much invalid data.
  8. **Pick & Rename Images:**
     - Aligns imagery dates with flux observation dates.
     - Copies matched chips into site-specific folders.
     - Renames files with a standardized convention:
        `HLS.{L30/S30}.{Tile}.{Date}.v002.{SiteID}.50x50pixels.tif`.

------

### **Supporting Utilities**

- Raster utilities: `raster2array`, `array2raster`, `gdal_merge_bands`.
- Distance calculation helpers.
- Uses **`lytools`** (custom package) for filesystem ops, multiprocessing, logging, etc.

------

### **Workflow Execution**

- `main()` runs:
  1. `Expand_points_to_rectangle().run()` → builds site rectangles.
  2. `Download_from_GEE().run()` → executes the entire download + preprocessing pipeline.

------

### **Output Structure**

- Organized directories with:
  - `sites.shp` (rectangles).
  - Downloaded raw zips.
  - Unzipped rasters.
  - Quality-controlled 50×50 image chips.
  - Multi-band merged TIFFs.
  - Final renamed datasets aligned with flux sites.

------

In short:
 This script automates the creation of **50×50 pixel quality-controlled HLS image chips** around flux tower sites, matched by observation dates, and prepares them in a standardized format for further analysis.