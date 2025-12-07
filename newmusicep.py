from deap import base, creator, tools
import random
import mido
import subprocess
import os

NOTE_RANGE = range(48, 84)  
DURATIONS = [120, 240, 480, 960]  # Added longer duration
VELOCITY = 64
INDIVIDUAL_LENGTH = 64  # Maybe could increase to 128 for longer songs. 

KEY_NOTES = {48, 50, 52, 53, 55, 57, 59, 
             60, 62, 64, 65, 67, 69, 71, 
             72, 74, 76, 77, 79, 81, 83}

def create_note():
    return {
        'pitch': random.choice(NOTE_RANGE),
        'duration': random.choice(DURATIONS),
        'velocity': VELOCITY
    }

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("note", create_note)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.note, n=INDIVIDUAL_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Enhanced Fitness Functions
def harmonic_fitness(individual):
    return sum(1 for note in individual if note['pitch'] in KEY_NOTES) / INDIVIDUAL_LENGTH,

def melodic_fitness(individual):
    score = 0
    for i in range(1, len(individual)):
        interval = abs(individual[i]['pitch'] - individual[i-1]['pitch'])
        # Penalize large jumps, reward stepwise motion
        if interval <= 2:
            score += 2
        elif interval <= 5:
            score += 1
        elif interval > 12:
            score -= 1
    return max(0, score) / INDIVIDUAL_LENGTH,

def rhythmic_fitness(individual):
    rhythm_pattern = [note['duration'] for note in individual]
    # Reward some repetition but not too much
    repeats = sum(1 for i in range(1, len(rhythm_pattern)) if rhythm_pattern[i] == rhythm_pattern[i-1])
    variety = len(set(rhythm_pattern))
    return (repeats / INDIVIDUAL_LENGTH + variety / len(DURATIONS)) / 2,

def contour_fitness(individual):
    # Reward interesting melodic contours
    pitches = [note['pitch'] for note in individual]
    changes = 0
    for i in range(2, len(pitches)):
        diff1 = pitches[i-1] - pitches[i-2]
        diff2 = pitches[i] - pitches[i-1]
        if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
            changes += 1
    return changes / (INDIVIDUAL_LENGTH - 2),

def range_fitness(individual):
    # Reward use of full range but penalize extremes
    pitches = [note['pitch'] for note in individual]
    pitch_range = max(pitches) - min(pitches)
    return min(pitch_range / 24, 1.0),

def total_fitness(individual):
    h_fit = harmonic_fitness(individual)[0]
    m_fit = melodic_fitness(individual)[0]
    r_fit = rhythmic_fitness(individual)[0]
    c_fit = contour_fitness(individual)[0]
    rg_fit = range_fitness(individual)[0]
    return (0.25 * h_fit + 
            0.25 * m_fit + 
            0.20 * r_fit + 
            0.15 * c_fit +
            0.15 * rg_fit),

toolbox.register("evaluate", total_fitness)

# Multiple Mutation Operators
def mutate_pitch(individual, indpb=0.15):
    """Mutate pitch values"""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i]['pitch'] = random.choice(NOTE_RANGE)
    return individual,

def mutate_duration(individual, indpb=0.15):
    """Mutate duration values"""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i]['duration'] = random.choice(DURATIONS)
    return individual,

def mutate_transpose(individual, indpb=0.1):
    """Transpose a section up or down"""
    if random.random() < indpb:
        start = random.randint(0, len(individual) - 8)
        end = start + random.randint(4, 16)
        transpose = random.choice([-12, -7, -5, 5, 7, 12])
        for i in range(start, min(end, len(individual))):
            new_pitch = individual[i]['pitch'] + transpose
            if new_pitch in NOTE_RANGE:
                individual[i]['pitch'] = new_pitch
    return individual,

def mutate_invert(individual, indpb=0.05):
    """Invert a melodic section"""
    if random.random() < indpb:
        start = random.randint(0, len(individual) - 8)
        end = start + random.randint(4, 12)
        pivot = individual[start]['pitch']
        for i in range(start, min(end, len(individual))):
            diff = individual[i]['pitch'] - pivot
            new_pitch = pivot - diff
            if new_pitch in NOTE_RANGE:
                individual[i]['pitch'] = new_pitch
    return individual,

def mutate_rhythm_shift(individual, indpb=0.1):
    """Shift rhythm pattern"""
    if random.random() < indpb:
        start = random.randint(0, len(individual) - 8)
        end = start + random.randint(4, 16)
        durations = [individual[i]['duration'] for i in range(start, min(end, len(individual)))]
        durations = durations[1:] + durations[:1]  # Rotate
        for i, dur in enumerate(durations):
            if start + i < len(individual):
                individual[start + i]['duration'] = dur
    return individual,

def mutate_comprehensive(individual):
    """Apply multiple mutation operators with different probabilities"""
    individual, = mutate_pitch(individual, indpb=0.1)
    individual, = mutate_duration(individual, indpb=0.1)
    individual, = mutate_transpose(individual, indpb=0.15)
    individual, = mutate_invert(individual, indpb=0.08)
    individual, = mutate_rhythm_shift(individual, indpb=0.12)
    return individual,

# Multiple Crossover Operators
def crossover_two_point(ind1, ind2):
    """Two-point crossover"""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, size - 1)
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = \
        ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def crossover_uniform(ind1, ind2, indpb=0.5):
    """Uniform crossover"""
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def crossover_segment(ind1, ind2):
    """Exchange random segments"""
    size = min(len(ind1), len(ind2))
    seg_length = random.randint(4, 16)
    start = random.randint(0, size - seg_length)
    ind1[start:start+seg_length], ind2[start:start+seg_length] = \
        ind2[start:start+seg_length], ind1[start:start+seg_length]
    return ind1, ind2

def crossover_interleave(ind1, ind2):
    """Interleave sections from both parents"""
    size = min(len(ind1), len(ind2))
    chunk_size = 4
    for i in range(0, size, chunk_size * 2):
        end = min(i + chunk_size, size)
        ind1[i:end], ind2[i:end] = ind2[i:end], ind1[i:end]
    return ind1, ind2

def apply_crossover(ind1, ind2):
    """Randomly select and apply a crossover operator"""
    crossover_type = random.random()
    if crossover_type < 0.3:
        return crossover_two_point(ind1, ind2)
    elif crossover_type < 0.6:
        return crossover_uniform(ind1, ind2, indpb=0.5)
    elif crossover_type < 0.85:
        return crossover_segment(ind1, ind2)
    else:
        return crossover_interleave(ind1, ind2)

toolbox.register("mate", apply_crossover)
toolbox.register("mutate", mutate_comprehensive)
toolbox.register("select", tools.selTournament, tournsize=4)

# MIDI Output
def individual_to_midi(individual, filename="output.mid"):
    # Create directory if it doesn't exist
    os.makedirs("midi_outputs", exist_ok=True)
    
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for note in individual:
        pitch = note['pitch']
        dur = note['duration']
        vel = note['velocity']
        
        track.append(mido.Message('note_on', note=pitch, velocity=vel, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=vel, time=dur))

    filepath = f"midi_outputs/{filename}"
    mid.save(filepath)
    return filepath

# Audio conversion function
def midi_to_audio(midi_file, output_file=None, soundfont=None):
    """Convert MIDI to WAV using fluidsynth"""
    if soundfont is None:
        # Try common soundfont locations
        possible_soundfonts = [
            os.path.expanduser("~/soundfonts/FluidR3_GM.sf2"),
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/local/share/soundfonts/default.sf2",
        ]
        soundfont = None
        for sf in possible_soundfonts:
            if os.path.exists(sf):
                soundfont = sf
                break
        
        if soundfont is None:
            print("⚠ Warning: No soundfont found. Audio conversion skipped.")
            print("  Download a soundfont:")
            print("  mkdir -p ~/soundfonts")
            print("  curl -L 'https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip' -o ~/soundfonts/FluidR3_GM.zip")
            print("  cd ~/soundfonts && unzip FluidR3_GM.zip")
            return None
    
    if output_file is None:
        output_file = midi_file.replace('.mid', '.wav').replace('midi_outputs', 'audio_outputs')
    
    # Create audio_outputs directory
    os.makedirs("audio_outputs", exist_ok=True)
    
    try:
        subprocess.run([
            'fluidsynth',
            '-ni',
            soundfont,
            midi_file,
            '-F',
            output_file,
            '-r',
            '44100'
        ], check=True, capture_output=True, text=True)
        print(f"  ✓ Audio: {output_file}")
        return output_file
    except FileNotFoundError:
        print("⚠ Warning: fluidsynth not installed. Run: brew install fluidsynth")
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting MIDI: {e.stderr}")
        return None
    except Exception as e:
        print(f"✗ Error converting MIDI: {e}")
        return None

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    population = toolbox.population(n=100)  # Increased population

    NGEN = 50  # Increased generations
    CXPB = 0.7  # Crossover probability
    MUTPB = 0.3  # Mutation probability
    
    print("Starting evolution...")
    print("=" * 60)
    
    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    for gen in range(NGEN):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

        # Statistics
        fits = [ind.fitness.values[0] for ind in population]
        best_gen = tools.selBest(population, 1)[0]
        
        if (gen + 1) % 10 == 0:
            print(f"\nGeneration {gen+1}:")
            print(f"  Best Fitness:  {best_gen.fitness.values[0]:.4f}")
            print(f"  Avg Fitness:   {sum(fits)/len(fits):.4f}")
            print(f"  Min Fitness:   {min(fits):.4f}")
            
            # Save MIDI
            midi_file = individual_to_midi(best_gen, filename=f"best_gen_{gen+1}.mid")
            print(f"  ✓ MIDI: {midi_file}")
            
            # Convert to audio
            midi_to_audio(midi_file)

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("=" * 60)
    
    # Final best individual
    top_ind = tools.selBest(population, 1)[0]
    print(f"\nFinal best fitness: {top_ind.fitness.values[0]:.4f}")
    
    midi_file = individual_to_midi(top_ind, filename="best_individual.mid")
    print(f"✓ Best MIDI: {midi_file}")
    midi_to_audio(midi_file)
    
    # Save top 5 individuals
    print("\nSaving top 5 individuals...")
    top_5 = tools.selBest(population, 5)
    for i, ind in enumerate(top_5):
        midi_file = individual_to_midi(ind, filename=f"top_{i+1}.mid")
        print(f"  Top {i+1} (fitness {ind.fitness.values[0]:.4f}): {midi_file}")
        midi_to_audio(midi_file)
    
    print("\n" + "=" * 60)
    print("All files saved!")
    print("  MIDI files: midi_outputs/")
    print("  Audio files: audio_outputs/")
    print("=" * 60)
    print("\nTo play audio files, run:")
    print("  open audio_outputs/best_individual.wav")
    print("Or just double-click the WAV files in Finder!")
