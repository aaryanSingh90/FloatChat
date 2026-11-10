"""
knowledge_base/documents.py
============================
Scientific documentation used as the RAG knowledge base.
Each document is a dict with keys: id, title, content, tags.
"""

DOCUMENTS = [
    # ------------------------------------------------------------------ Argo
    {
        "id": "argo_overview",
        "title": "Argo Profiling Floats – Overview",
        "tags": ["argo", "float", "profiling", "overview"],
        "content": (
            "Argo is an international program that deploys autonomous profiling floats throughout "
            "the world's oceans. Each float drifts at a parking depth (typically 1000 dbar), "
            "descends to a profile depth (typically 2000 dbar), and then rises while measuring "
            "temperature, salinity, and pressure. Data are transmitted to shore via satellite "
            "within hours of surfacing. As of 2024 the global array holds more than 4,000 "
            "active floats producing roughly 400 profiles per day. Float IDs follow the WMO "
            "numbering scheme (e.g. WMO_1234567)."
        ),
    },
    {
        "id": "argo_data_cycle",
        "title": "Argo Float Data Cycle",
        "tags": ["argo", "float", "cycle", "surfacing", "parking depth"],
        "content": (
            "An Argo float follows a 10-day cycle. It descends from the surface to the parking "
            "depth (~1000 m), drifts for approximately 9 days, then descends further to the "
            "profiling depth (~2000 m) before ascending. During the ascent it records a "
            "continuous CTD profile. At the surface it transmits data via Argos satellite or "
            "Iridium and receives new mission parameters. Each vertical profile is called a "
            "float cycle."
        ),
    },
    {
        "id": "argo_float_types",
        "title": "Argo Float Hardware Types",
        "tags": ["argo", "float", "hardware", "APEX", "SOLO", "NAVIS", "ARVOR"],
        "content": (
            "Common Argo float models include: APEX (Autonomous Profiling Explorer) by Webb "
            "Research; SOLO by Scripps Institution of Oceanography; NAVIS by Sea-Bird Scientific; "
            "ARVOR by IFREMER/NKE. BGC-Argo floats carry additional biogeochemical sensors "
            "measuring oxygen, nitrate, chlorophyll-a, backscatter, irradiance, and pH. "
            "Deep Argo floats profile to 6000 dbar."
        ),
    },
    # ----------------------------------------------------------- Temperature
    {
        "id": "temperature_overview",
        "title": "Sea-Water Temperature in Oceanography",
        "tags": ["temperature", "CTD", "thermocline", "in-situ"],
        "content": (
            "Sea-water temperature is measured in degrees Celsius (°C). Argo floats measure "
            "in-situ temperature (ITS-90 scale). Surface temperatures range from about -1.8 °C "
            "in polar seas to >30 °C in tropical regions. A sharp thermocline – a rapid decrease "
            "in temperature with depth – typically lies between 100 m and 500 m. Below the "
            "thermocline, deep ocean temperature is relatively uniform, often 1–4 °C. Temperature "
            "is a primary driver of ocean density and circulation."
        ),
    },
    {
        "id": "temperature_anomaly",
        "title": "Sea Surface Temperature Anomaly (SSTA)",
        "tags": ["temperature", "anomaly", "SST", "El Nino", "climate"],
        "content": (
            "Sea Surface Temperature Anomaly (SSTA) is the deviation of observed SST from the "
            "long-term climatological mean. Positive SSTA (warmer-than-average water) is "
            "associated with El Niño events in the Pacific. Persistent positive SSTA in the "
            "North Atlantic has been linked to intensified hurricane activity. Argo float data "
            "are routinely assimilated into global climate monitoring systems to track SSTA."
        ),
    },
    # --------------------------------------------------------------- Salinity
    {
        "id": "salinity_overview",
        "title": "Ocean Salinity – Practical Salinity Units (PSU)",
        "tags": ["salinity", "PSU", "PSS-78", "halocline"],
        "content": (
            "Salinity quantifies the dissolved salt content of sea water. The 1978 Practical "
            "Salinity Scale (PSS-78) defines Practical Salinity Units (PSU). Open-ocean salinity "
            "typically ranges from 33 to 37 PSU. High values (>38 PSU) occur in the "
            "Mediterranean and Red Sea due to high evaporation. Low values (<31 PSU) occur near "
            "river mouths, sea-ice melt zones, and the Arctic Ocean. A halocline is a sharp "
            "vertical gradient in salinity, common beneath sea-ice and near river plumes."
        ),
    },
    {
        "id": "salinity_density",
        "title": "Salinity and Ocean Density",
        "tags": ["salinity", "density", "thermohaline", "circulation"],
        "content": (
            "Ocean density depends on temperature and salinity (and pressure). Cold, salty water "
            "is denser and sinks, driving the global thermohaline circulation (also called the "
            "Ocean Conveyor Belt). The North Atlantic Deep Water (NADW) forms when warm, salty "
            "Gulf Stream water cools and sinks near Greenland and Iceland. Changes in salinity "
            "due to freshwater influx from melting ice sheets can weaken this circulation."
        ),
    },
    # --------------------------------------------------------------- Pressure
    {
        "id": "pressure_depth",
        "title": "Pressure and Depth in Oceanography",
        "tags": ["pressure", "depth", "dbar", "decibars", "CTD"],
        "content": (
            "Ocean pressure is measured in decibars (dbar). One decibar ≈ 1 metre of seawater "
            "depth (more precisely, depth(m) ≈ pressure(dbar) × 0.995 at mid-latitudes). Argo "
            "floats measure absolute pressure with a precision of ±2.4 dbar. The standard "
            "profiling depth for Argo is 2000 dbar (~2000 m). Pressure data are used to convert "
            "in-situ temperature to potential temperature and to calculate derived variables "
            "such as density and buoyancy frequency."
        ),
    },
    {
        "id": "depth_zones",
        "title": "Ocean Depth Zones",
        "tags": ["depth", "epipelagic", "mesopelagic", "bathypelagic", "abyssal"],
        "content": (
            "The ocean is divided into depth zones: "
            "Epipelagic (sunlit zone): 0–200 m – most biological activity and heat absorption. "
            "Mesopelagic (twilight zone): 200–1000 m – decreasing light; thermocline present. "
            "Bathypelagic (midnight zone): 1000–4000 m – no sunlight; near-constant temperature. "
            "Abyssal zone: 4000–6000 m – extreme pressure; sparse but specialised life. "
            "Hadal zone: >6000 m – ocean trenches only."
        ),
    },
    # ------------------------------------------------------------------ QC
    {
        "id": "qc_flags",
        "title": "Argo Quality Control (QC) Flags",
        "tags": ["QC", "quality control", "flag", "data quality"],
        "content": (
            "Argo data use a standard set of QC flags: "
            "0 = No QC performed. "
            "1 = Good data. "
            "2 = Probably good data. "
            "3 = Probably bad data – adjusted if possible. "
            "4 = Bad data – do not use. "
            "5 = Value changed during delayed-mode QC. "
            "8 = Estimated value (e.g. from interpolation). "
            "9 = Missing value. "
            "Scientific analyses should use only flag 1 and 2 data. Flags are applied to each "
            "individual parameter (temperature, salinity, pressure) independently."
        ),
    },
    {
        "id": "qc_real_time_delayed",
        "title": "Real-Time vs Delayed-Mode QC",
        "tags": ["QC", "RTQC", "DMQC", "delayed mode"],
        "content": (
            "Argo data pass through two QC stages. Real-Time QC (RTQC) is applied automatically "
            "within hours of data transmission. It performs range checks, spike detection, "
            "gradient tests, and density inversion checks. Delayed-Mode QC (DMQC) is applied "
            "by expert oceanographers months later; it uses climatological comparisons and "
            "statistical methods (e.g. OWC) to detect sensor drift and apply salinity "
            "corrections. DMQC data are stored in D-files and are the most reliable."
        ),
    },
    # ---------------------------------------------------------------- Schema
    {
        "id": "dataset_schema",
        "title": "Float Chat Dataset Schema",
        "tags": ["schema", "dataset", "columns", "fields"],
        "content": (
            "The Float Chat dataset contains the following columns: "
            "float_id (string): WMO float identifier. "
            "latitude (float): degrees North, -90 to +90. "
            "longitude (float): degrees East, -180 to +180. "
            "date (string): observation date in YYYY-MM-DD format. "
            "temperature (float): in-situ temperature in °C. "
            "salinity (float): practical salinity in PSU. "
            "pressure (float): pressure in decibars (dbar). "
            "depth (float): approximate depth in metres. "
            "region (string): named ocean region (e.g. 'North Atlantic'). "
            "qc_flag (int): Argo QC flag (1=good … 4=bad)."
        ),
    },
    # ------------------------------------------------------------ Regions
    {
        "id": "ocean_regions",
        "title": "Major Ocean Regions in Float Chat",
        "tags": ["region", "ocean basin", "Atlantic", "Pacific", "Indian", "Southern", "Arctic"],
        "content": (
            "Float Chat covers ten ocean regions: "
            "North Atlantic (30–60°N, 60–10°W) – Gulf Stream, thermohaline circulation. "
            "South Atlantic (50–10°S, 50°W–10°E) – Benguela current, Brazil current. "
            "North Pacific (20–55°N, 140°E–120°W) – Kuroshio, North Pacific Gyre. "
            "South Pacific (60–10°S, 170–70°W) – Antarctic Circumpolar Current, Humboldt. "
            "Indian Ocean (40°S–25°N, 40–100°E) – monsoon-driven circulation. "
            "Southern Ocean (70–50°S) – strongest winds; principal carbon sink. "
            "Mediterranean Sea (30–46°N, 5°W–36°E) – high salinity; WMDW formation. "
            "Arctic Ocean (70–85°N) – rapidly changing; ice-influenced. "
            "Caribbean Sea (10–25°N, 85–60°W) – warm tropical water; hurricane genesis region. "
            "Coral Sea (30–10°S, 145–165°E) – Great Barrier Reef adjacent; high biodiversity."
        ),
    },
    # -------------------------------------------------- Scientific terms
    {
        "id": "thermohaline_circulation",
        "title": "Thermohaline Circulation (THC)",
        "tags": ["thermohaline", "circulation", "AMOC", "conveyor belt", "deep water"],
        "content": (
            "The thermohaline circulation (THC) is the global system of ocean currents driven by "
            "density differences caused by temperature (thermo) and salinity (haline). It is often "
            "called the global ocean conveyor belt. The Atlantic component (AMOC – Atlantic "
            "Meridional Overturning Circulation) transports heat northward and is critical to "
            "European climate. Argo data are essential for monitoring AMOC strength by tracking "
            "water mass properties."
        ),
    },
    {
        "id": "mixed_layer_depth",
        "title": "Ocean Mixed Layer and Mixed Layer Depth (MLD)",
        "tags": ["mixed layer", "MLD", "turbulence", "wind mixing", "surface"],
        "content": (
            "The ocean mixed layer is the surface layer where wind stress and convective cooling "
            "produce vigorous mixing, resulting in nearly uniform temperature and salinity. The "
            "Mixed Layer Depth (MLD) is defined as the depth at which density (or temperature) "
            "first deviates significantly from the surface value (commonly Δσ = 0.03 kg m⁻³ or "
            "ΔT = 0.2 °C). MLD is seasonally variable: deep in winter due to cooling, shallow in "
            "summer due to warming. Argo profiles are widely used to compute global MLD climatologies."
        ),
    },
    {
        "id": "potential_density",
        "title": "Potential Temperature and Potential Density",
        "tags": ["potential temperature", "potential density", "sigma-theta", "adiabatic"],
        "content": (
            "Potential temperature (θ) is the temperature a parcel of water would have if moved "
            "adiabatically to a reference pressure (usually surface). It removes the effect of "
            "compression, allowing comparison of water masses at different depths. Potential "
            "density (σθ) is calculated using θ and salinity referenced to surface pressure. "
            "σθ = density − 1000 kg m⁻³ for convenience. Water masses are traced by their "
            "characteristic θ-S (temperature-salinity) relationship."
        ),
    },
    {
        "id": "southern_ocean",
        "title": "Southern Ocean – Key Oceanographic Features",
        "tags": ["Southern Ocean", "ACC", "carbon sink", "Antarctic", "circumpolar"],
        "content": (
            "The Southern Ocean encircles Antarctica and plays a disproportionate role in global "
            "climate. The Antarctic Circumpolar Current (ACC) is the world's largest ocean "
            "current, flowing eastward around Antarctica. The Southern Ocean absorbs about 40% of "
            "anthropogenic CO₂ and 70% of the excess heat stored by Earth. Argo floats in the "
            "Southern Ocean face harsh conditions: sea ice, strong storms, and remote locations. "
            "Temperature near the surface ranges from −1.9 °C in winter to ~4 °C in summer."
        ),
    },
    {
        "id": "query_operations",
        "title": "Supported Query Operations in Float Chat",
        "tags": ["query", "operations", "average", "maximum", "minimum", "count", "sum", "filter"],
        "content": (
            "Float Chat supports the following analytical operations: "
            "AVERAGE (avg, mean): arithmetic mean of a numeric column. "
            "MAXIMUM (max): highest value in a column or group. "
            "MINIMUM (min): lowest value in a column or group. "
            "COUNT: number of observations matching filters. "
            "SUM: total of a numeric column. "
            "MEDIAN: middle value of a numeric column. "
            "GROUP BY: aggregate by region, year, month, or qc_flag. "
            "FILTER: restrict to specific region, date range, depth range, or QC flag. "
            "SORT: order results by any column ascending or descending. "
            "COMPARE: side-by-side aggregation across regions."
        ),
    },
]

