use mixer::mix;
use rand::Rng;

#[test]
fn test_add() {
    let mut rng = rand::thread_rng();
    let mut sample_buffer = [0f64; 200];
    rng.fill(&mut sample_buffer[..]);
    let carrier_frequency = 3e3;
    assert_eq!(mix(&sample_buffer, carrier_frequency),());
} 