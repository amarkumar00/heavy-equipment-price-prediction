PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS equipment_transactions;
DROP TABLE IF EXISTS equipment_attributes;

CREATE TABLE equipment_attributes (
    asset_id TEXT PRIMARY KEY,
    manufacture_year REAL,
    spec_full_descriptor TEXT,
    spec_base_class TEXT,
    spec_sub_class TEXT,
    spec_release_series TEXT,
    spec_variant_modifier TEXT,
    asset_scale_factor TEXT,
    functional_classification TEXT,
    inventory_group_category TEXT,
    inventory_group_description TEXT,
    cabin_type TEXT,
    forks TEXT,
    drivetrain_type TEXT,
    product_config_id TEXT,
    vendor_partner_id TEXT
);

CREATE TABLE equipment_transactions (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    transaction_date TEXT,
    target_value REAL NOT NULL CHECK (target_value > 0),
    operational_hours_meter REAL,
    region_code TEXT,
    data_origin_code TEXT,
    utilization_tier TEXT,
    age REAL,
    log_operational_hours REAL,
    hours_per_year REAL,
    operational_hours_missing INTEGER NOT NULL CHECK (operational_hours_missing IN (0, 1)),
    operational_hours_zero INTEGER NOT NULL CHECK (operational_hours_zero IN (0, 1)),
    FOREIGN KEY (asset_id) REFERENCES equipment_attributes(asset_id)
);

CREATE INDEX idx_transactions_asset
    ON equipment_transactions(asset_id);
CREATE INDEX idx_transactions_date
    ON equipment_transactions(transaction_date);
CREATE INDEX idx_transactions_region
    ON equipment_transactions(region_code);
CREATE INDEX idx_attributes_inventory_category
    ON equipment_attributes(inventory_group_category);
