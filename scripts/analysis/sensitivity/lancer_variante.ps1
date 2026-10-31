# Consigne 012 : lance la chaine d'analyse pour une variante.
# Usage : powershell -File lancer_variante.ps1 -Exclure "B11_DixonPrice" [-Etapes etape5,p1,...]
# -Exclure "" = variante sans exclusion (controle de reproduction de 009e).
param(
    [string]$Exclure = "",
    [string[]]$Etapes = @("etape5_statistiques", "p1_succes_156_et_comparaison_v04", "tl1_ntot_15paires",
                          "b1_seuil_noeud2", "b2_arbre_decision_temps", "b3_validation_LOO", "b4_pareto_temps",
                          "p5_multistart_G3")
)
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:SENS_EXCLUDE = $Exclure
$nom = if ($Exclure) { "excl_" + ($Exclure -replace ",", "_") } else { "aucune_exclusion" }
# Depot publie (v0.5.2) : memes sorties que paths.py (ANALYSIS_OUT, defaut analysis_output/ a la racine du depot).
$base = if ($env:ANALYSIS_OUT) { $env:ANALYSIS_OUT } else { Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $here))) "analysis_output" }
$out = Join-Path (Join-Path $base "sensitivity") $nom
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $here
try {
    foreach ($e in $Etapes) {
        $t0 = Get-Date
        & python -X utf8 "$e.py" *> (Join-Path $out "$e`_console.txt")
        $code = $LASTEXITCODE
        $dt = ((Get-Date) - $t0).TotalSeconds
        "{0} | {1,-36} | code {2} | {3,7:N1} s" -f $nom, $e, $code, $dt | Tee-Object -FilePath (Join-Path $out "_journal.txt") -Append
        if ($code -ne 0) { throw "echec de $e (code $code) : voir $e`_console.txt" }
    }
} finally { Pop-Location; Remove-Item Env:\SENS_EXCLUDE -ErrorAction SilentlyContinue }
