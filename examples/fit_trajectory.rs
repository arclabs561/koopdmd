//! Fit consecutive snapshots from stdin and predict the final, held-out state.
use koopdmd::{dmd, SnapshotMatrix};
use std::error::Error;
use std::io::{self, BufRead, Write};

fn report(input: impl BufRead, mut output: impl Write) -> Result<(), Box<dyn Error>> {
    let mut states = Vec::new();
    for (index, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let state = line
            .split_whitespace()
            .map(|field| {
                field
                    .parse::<f64>()
                    .map_err(|_| format!("line {}: expected a number, got {field:?}", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        states.push(state);
    }

    if states.len() < 3 {
        return Err("expected at least three snapshots, including one held-out row".into());
    }
    let (held_out, train) = states.split_last().unwrap();
    let model = dmd(&SnapshotMatrix::new(train)?)?;
    let previous = train.last().unwrap();
    if held_out.len() != previous.len() || held_out.iter().any(|x| !x.is_finite()) {
        return Err("held-out row must contain one finite number per state component".into());
    }
    for value in model.eigenvalues() {
        writeln!(output, "eigenvalue {value}")?;
    }
    writeln!(output, "component predicted observed")?;
    for (i, (row, observed)) in model.operator().iter().zip(held_out).enumerate() {
        let predicted: f64 = row.iter().zip(previous).map(|(a, x)| a * x).sum();
        writeln!(output, "{i} {predicted:.6} {observed:.6}")?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    report(io::stdin().lock(), io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_observation_does_not_influence_fit() {
        let mut out = Vec::new();
        report(&b"1\n2\n4\n100\n"[..], &mut out).unwrap();
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("0 8.000000 100.000000"));
    }

    #[test]
    fn rejects_invalid_held_out_rows() {
        for input in ["1\n2\n3 4\n", "1\n2\nNaN\n", "1\n2\n"] {
            assert!(report(input.as_bytes(), io::sink()).is_err());
        }
    }
}
