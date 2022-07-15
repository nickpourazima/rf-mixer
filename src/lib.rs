use convolve2d::{convolve2d, DynamicMatrix};
use ndarray::Array1;
use num_complex::Complex;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::f64::consts::PI;
use std::f64::consts::SQRT_2;
use log::info;
use env_logger;

pub fn mix(sample_buffer: &[f64], carrier_frequency: f64)
// This function mixes the baseband samples in `sample_buffer` to the desired `carrier_frequency`.
{
    // Setup logging
    env_logger::init();
    
    // TODO: Input Sanitization

    // Convert input stream to complex array
    let buffer: &[Complex<f64>] = convert_to_complex(sample_buffer);

    // Formulate root-raised cosine function, store coefficients
    let n: usize = buffer.len().try_into().unwrap();
    let ts: f64 = 1e-3;
    let fs: f64 = 6e4;
    // Used for plotting
    let _bw: f64 = 1.0 / (2.0 * ts);
    let _ups: i64 = (ts * fs) as i64;
    // Filter parameters
    let alpha: f64 = 1.0;
    let n_filter: usize = usize::try_from((2.0 * 3.0 * ts * fs) as i32).unwrap();

    // Set the RRC filter to span 3 baseband samples to the left/right, introduce a delay of 3.0 * ts seconds
    let (time_idx, coefficients) = rrcosfilter(n_filter, alpha, 3.0 * ts, fs);
    info!(
        "Buffer: {:#?}\nCarrier Frequency: {}\nTime: {:#?}, Coefficients: {:#?}",
        buffer, carrier_frequency, time_idx, coefficients
    );

    // Convolve input with filter
    let output = convolve2d(
        &DynamicMatrix::new(n, 1, buffer.to_vec()).unwrap(),
        &DynamicMatrix::new(n_filter, 1, coefficients.to_vec()).unwrap(),
    );
    info!("Convolution: {:#?}", output);

    // Split into real & imag components (IQ)
    let (_,_,parsed_output) = (output).into_parts();
    let mut i: Vec<f64> = vec![0.0 ; parsed_output.len()/2];
    let mut q: Vec<f64> = vec![0.0 ; parsed_output.len()/2];
    let mut ix: usize = 0;
    for val in parsed_output.iter() {
        if ix == parsed_output.len()/2 {
            break;
        }
        else {        
            i[ix] = val.re;
            q[ix] = val.im;
            ix += 1;
        }
    }
    info!("I: {:#?}\nQ: {:#?}",i,q);
    
    // Up-conversion: multiply I/Q by cosine/sine
    let t_u = (Array1::<f64>::range(0.0, parsed_output.len() as f64, 1.0))/fs;
    info!("t_u: {:#?}", t_u);
    let constants = 2.0 * PI * carrier_frequency;
    let mut new_c = t_u.to_vec();
    let mut new_s = t_u.to_vec();
    new_c.iter_mut().for_each(|x| *x = (*x*constants).cos());
    new_s.iter_mut().for_each(|x| *x = (*x*-constants).sin());    

    let i_up : Vec<f64> = i.iter().zip(new_c).map(|(a, b)| a * b).collect();
    let q_up : Vec<f64> = q.iter().zip(new_s).map(|(x, y)| x * y).collect();
    info!("I_UP: {:#?}\nQ_UP: {:#?}", i_up, q_up);

    // Return summation as upconverted IQ
    let s_up : Vec<f64> = i_up.iter().zip(q_up).map(|(a, b)| a + b).collect();
    info!("S_UP: {:#?}", s_up);
}
fn convert_to_complex(buffer: &[f64]) -> &[Complex<f64>] {
    unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *mut Complex<f64>, buffer.len() / 2) }
}

fn rrcosfilter(n: usize, alpha: f64, ts: f64, fs: f64) -> (Array1<f64>, Array1<f64>) {
    //    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    //    Implementation based on CommPy's rrcosfilter: https://github.com/veeresht/CommPy/blob/master/commpy/filters.py
    //    Parameters
    //    ----------
    //    n : int
    //        Length of the filter in samples.
    //    alpha : float
    //        Roll off factor (Valid values are [0, 1]).
    //    ts : float
    //        Symbol period in seconds.
    //    fs : float
    //        Sampling Rate in Hz.
    //
    //    Returns
    //    ---------
    //    time_idx : 1-D ndarray of f64
    //        Array containing the time indices, in seconds, for
    //        the impulse response.
    //    h_rrc : 1-D ndarray of f64
    //        Impulse response of the root raised cosine filter.
    let t_delta = 1.0 / (fs);
    let time_idx = (Array1::<f64>::range(0.0, n as f64, 1.0) - n as f64 / 2.0) * t_delta;
    let sample_num = Array1::<f64>::range(0.0, n as f64, 1.0);
    let mut h_rrc: Vec<f64> = vec![0.0; n];

    for x in sample_num {
        let t = (x - n as f64 / 2.0) * t_delta;
        if t == 0.0 {
            h_rrc[x as usize] = 1.0 - alpha + (4.0 * alpha / PI);
        } else if alpha != 0.0 && (t == ts / (4.0 * alpha) || t == -ts / (4.0 * alpha)) {
            h_rrc[x as usize] = (alpha / SQRT_2)
                * ((1.0 + (2.0 / PI)) * (PI / (4.0 * alpha)).sin())
                + ((1.0 - (2.0 / PI)) * (PI / (4.0 * alpha)).cos());
        } else {
            h_rrc[x as usize] = (((PI * t * (1.0 - alpha)) / ts).sin()
                + 4.0 * alpha * (t / ts) * ((PI * t * (1.0 + alpha)) / ts).cos())
                / (PI * t * (1.0 - (4.0 * alpha * t / ts) * (4.0 * alpha * t / ts)) / ts);
        }
    }
    (time_idx, h_rrc.try_into().unwrap())
}
