from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pandas as pd
import os
import json
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, Player, CSVLineupExporter
import tempfile

app = Flask(__name__)
app.secret_key = 'mlb_dfs_optimizer_secret_key'

# Constants
SALARY_CAP = 35000
REQUIRED_PLAYERS = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
TOTAL_PLAYERS = 9

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_file_path)
        
        try:
            # Process the file (clean data)
            df = pd.read_csv(temp_file_path)
            
            # Basic cleaning (similar to data_cleaning.py)
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Save the cleaned data
            cleaned_file_path = 'cleaned_players_list.csv'
            df.to_csv(cleaned_file_path, index=False)
            
            flash(f'File uploaded and processed successfully!')
            return redirect(url_for('index'))
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        # Get parameters from form
        num_lineups = int(request.form.get('num_lineups', 10))
        max_exposure = float(request.form.get('max_exposure', 1.0))
        min_salary = int(request.form.get('min_salary', 0))
        max_salary = int(request.form.get('max_salary', SALARY_CAP))
        
        # Get player exclusions/locks if any
        excluded_players = request.form.get('excluded_players', '').split(',')
        excluded_players = [p.strip() for p in excluded_players if p.strip()]
        
        locked_players = request.form.get('locked_players', '').split(',')
        locked_players = [p.strip() for p in locked_players if p.strip()]
        
        # Initialize the optimizer
        optimizer = get_optimizer(Site.FANDUEL, Sport.BASEBALL)
        
        # Load players from CSV
        optimizer.load_players_from_csv('cleaned_players_list.csv')
        
        # Apply constraints
        if min_salary > 0:
            optimizer.set_min_salary_cap(min_salary)
        
        if max_salary < SALARY_CAP:
            optimizer.set_max_salary_cap(max_salary)
        
        # Handle player exclusions
        for player_id in excluded_players:
            player = optimizer.get_player_by_id(player_id)
            if player:
                optimizer.remove_player(player)
        
        # Handle player locks
        for player_id in locked_players:
            player = optimizer.get_player_by_id(player_id)
            if player:
                optimizer.add_player_to_lineup(player)
        
        # Set max exposure if needed
        if max_exposure < 1.0:
            for player in optimizer.players:
                optimizer.set_player_max_exposure(player, max_exposure)
        
        # Generate lineups
        lineups = optimizer.optimize(n=num_lineups)
        
        # Export lineups to CSV
        exporter = CSVLineupExporter(lineups)
        export_file = 'exported_lineups.csv'
        exporter.export(export_file)
        
        # Convert lineups to JSON for display
        lineup_data = []
        for lineup in lineups:
            lineup_players = []
            for player in lineup.players:
                lineup_players.append({
                    'id': player.id,
                    'name': f"{player.first_name} {player.last_name}",
                    'position': player.position,
                    'team': player.team,
                    'opponent': player.opponent,
                    'salary': player.salary,
                    'fppg': player.fppg
                })
            
            lineup_data.append({
                'players': lineup_players,
                'total_salary': lineup.salary_costs,
                'total_fppg': lineup.fantasy_points_projection
            })
        
        # Store in session for display
        session['lineups'] = lineup_data
        
        return redirect(url_for('results'))
    
    except Exception as e:
        flash(f'Error generating lineups: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    lineups = session.get('lineups', [])
    return render_template('results.html', lineups=lineups)

@app.route('/players')
def get_players():
    try:
        df = pd.read_csv('cleaned_players_list.csv')
        players = df.to_dict('records')
        return jsonify(players)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
