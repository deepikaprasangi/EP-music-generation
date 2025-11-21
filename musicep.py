from deap import base, creator, tools
import random
import mido

NOTE_RANGE = range(48, 84)  
DURATIONS = [120, 240, 480] 
VELOCITY = 64
INDIVIDUAL_LENGTH = 16

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

# Build the Fitness Function

def harmonic_fitness(individual):
    return sum(1 for note in individual if note['pitch'] in KEY_NOTES) / INDIVIDUAL_LENGTH,

def melodic_fitness(individual):
    score = 0
    for i in range(1, len(individual) - 1):
        interval = abs(individual[i]['pitch'] - individual[i-1]['pitch'])
        score += max(0, 12 - interval)
    return score / (INDIVIDUAL_LENGTH - 2),

def rhythmic_fitness(individual):
    rhythm_pattern = [note['duration'] for note in individual]
    repeats = sum(1 for i in range(1, len(rhythm_pattern)) if rhythm_pattern[i] == rhythm_pattern[i-1])
    return repeats / (INDIVIDUAL_LENGTH - 1),

def musicality_fitness(individual):
    pitches = [note['pitch'] for note in individual]
    return len(set(pitches)) / (INDIVIDUAL_LENGTH),

def total_fitness(individual):
    h_fit = harmonic_fitness(individual)[0]
    m_fit = melodic_fitness(individual)[0]
    r_fit = rhythmic_fitness(individual)[0]
    mu_fit = musicality_fitness(individual)[0]
    return (0.30 * h_fit + 
            0.25 * m_fit + 
            0.25 * r_fit + 
            0.20 * mu_fit),

toolbox.register("evaluate", total_fitness)

def mutate_note(note):
    if random.random() < 0.3:
        note['pitch'] = random.choice(NOTE_RANGE)
    if random.random() < 0.3:
        note['duration'] = random.choice(DURATIONS)
    return note,

def mutate_individual(individual):
    for i in range(len(individual)):
        if random.random() < 0.2:
            individual[i], = mutate_note(individual[i])
    return individual,

toolbox.register("mate", tools.cxOnePoint)

toolbox.register("mutate", mutate_individual)

toolbox.register("select", tools.selTournament, tournsize=3)

# Get the MIDI Output

def individual_to_midi(individual, filename="output.mid"):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for note in individual:
        pitch = note['pitch']
        dur = note['duration']
        vel = note['velocity']
        
        track.append(mido.Message('note_on', note=pitch, velocity=vel, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=vel, time=dur))

    mid.save(f"midi_outputs/{filename}")

if __name__ == "__main__":
    population = toolbox.population(n=50)

    NGEN = 40
    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        best_gen = tools.selBest(population, 1)[0]
        individual_to_midi(best_gen, filename=f"best_gen_{gen+1}.mid")
        print(f"Generation {gen+1}: Best fitness = {best_gen.fitness.values[0]}")

    top_ind = tools.selBest(population, 1)[0]
    print("Best individual fitness:", top_ind.fitness.values[0])
    individual_to_midi(top_ind, filename="best_individual.mid")