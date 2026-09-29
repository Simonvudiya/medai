const fs = require('fs');

// Load compounds data
const compoundsData = JSON.parse(fs.readFileSync('present_compounds.txt', 'utf8'));

// Read template
let html = fs.readFileSync('template_compounds.html', 'utf8');

// Replace placeholder with actual data
html = html.replace('[COMPOUNDS_DATA]', JSON.stringify(compoundsData.compounds));
html = html.replace('[PLANTS_MENTIONED]', JSON.stringify(compoundsData.plantsMentioned));
html = html.replace('[REPORT_INFO]', JSON.stringify({
    report: compoundsData.report,
    researcher: compoundsData.researcher,
    description: compoundsData.description
}));

fs.writeFileSync('compounds.html', html);
console.log('Created compounds.html with ' + compoundsData.compounds.length + ' compounds');