// Main JavaScript file for the MLB DFS Lineup Optimizer

// Show loading spinner
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Loading...</p>
            </div>
        `;
    }
}

// Format currency
function formatCurrency(amount) {
    return '$' + parseFloat(amount).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format decimal
function formatDecimal(value, decimals = 1) {
    return parseFloat(value).toFixed(decimals);
}

// Add animation class to elements
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in class to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.classList.add('fade-in');
    });
    
    // Form validation
    const optimizerForm = document.getElementById('optimizerForm');
    if (optimizerForm) {
        optimizerForm.addEventListener('submit', function(event) {
            // Basic validation
            const numLineups = document.getElementById('num_lineups').value;
            const minSalary = document.getElementById('min_salary').value;
            const maxSalary = document.getElementById('max_salary').value;
            
            if (parseInt(minSalary) > parseInt(maxSalary)) {
                event.preventDefault();
                alert('Minimum salary cannot be greater than maximum salary!');
            }
            
            if (parseInt(numLineups) < 1 || parseInt(numLineups) > 150) {
                event.preventDefault();
                alert('Number of lineups must be between 1 and 150!');
            }
        });
    }
});

// Toggle player selection in the player list
function togglePlayerSelection(playerId, action) {
    const playerItem = document.querySelector(`.player-item[data-id="${playerId}"]`);
    
    if (!playerItem) return;
    
    if (action === 'exclude') {
        playerItem.classList.toggle('bg-light');
        
        // Update excluded players input
        const excludedPlayersInput = document.getElementById('excluded_players');
        let excludedPlayers = excludedPlayersInput.value.split(',').map(id => id.trim()).filter(id => id);
        
        if (playerItem.classList.contains('bg-light')) {
            // Add to excluded
            if (!excludedPlayers.includes(playerId)) {
                excludedPlayers.push(playerId);
            }
        } else {
            // Remove from excluded
            excludedPlayers = excludedPlayers.filter(id => id !== playerId);
        }
        
        excludedPlayersInput.value = excludedPlayers.join(',');
    } else if (action === 'lock') {
        playerItem.classList.toggle('bg-info');
        playerItem.classList.toggle('bg-opacity-25');
        
        // Update locked players input
        const lockedPlayersInput = document.getElementById('locked_players');
        let lockedPlayers = lockedPlayersInput.value.split(',').map(id => id.trim()).filter(id => id);
        
        if (playerItem.classList.contains('bg-info')) {
            // Add to locked
            if (!lockedPlayers.includes(playerId)) {
                lockedPlayers.push(playerId);
            }
        } else {
            // Remove from locked
            lockedPlayers = lockedPlayers.filter(id => id !== playerId);
        }
        
        lockedPlayersInput.value = lockedPlayers.join(',');
    }
}
