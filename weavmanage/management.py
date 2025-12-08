import os
import json
import logging
import importlib.util

# Migrations directory
MIGRATIONS_DIR = "migrations"

# Path to the applied migrations tracking file
APPLIED_MIGRATIONS_FILE = f"/app/active/migrations.json"

def get_applied_migrations():
    """Get the list of applied migrations"""
    if os.path.exists(APPLIED_MIGRATIONS_FILE):
        with open(APPLIED_MIGRATIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_applied_migrations(migrations):
    """Save applied migrations"""
    with open(APPLIED_MIGRATIONS_FILE, "w") as f:
        json.dump(migrations, f)

def import_migration_script(script_name):
    """Dynamically import a migration script from the migrations folder"""
    script_path = os.path.join(MIGRATIONS_DIR, script_name)
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    migration_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration_module)
    return migration_module

def run_migrations(client):
    """Run all migrations"""
    applied_migrations = get_applied_migrations()

    # Get all migration scripts from the migrations directory, sorted by filename
    migration_files = sorted(os.listdir(MIGRATIONS_DIR))

    for migration_file in migration_files:
        if migration_file.endswith(".py"):
            migration_version = migration_file.split('_')[0]  # Assuming version is the first part of the filename (e.g., '001')
            
            # Skip migrations that have already been applied
            if migration_version in applied_migrations:
                logging.debug(f"Migration {migration_version} already applied, skipping.")
                continue
            
            logging.debug(f"Running migration {migration_version}...")
            
            # Import and run the migration script
            try:
                migration_module = import_migration_script(migration_file)
            
            # Check if the migration module has a `run` function and execute it
                if hasattr(migration_module, 'run'):
                    migration_module.run(client)
                applied_migrations.append(migration_version)
                save_applied_migrations(applied_migrations)

            except Exception as e:
                logging.error(f"Error running migration {migration_version}: {e}")
                continue