"""Inspectable contextual action values for isolated simulations, not moral scores."""
from copy import deepcopy
import json
import math
import sqlite3

VERSION = 'contextual-action-policy/0.1'


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ContextualPolicy:
    """One linear value model per action; no scenario IDs or text as features.

    Normalized incremental squared-error updates generalize through shared cues
    and their pairwise interactions. Softmax values are selection probabilities,
    never calibrated probabilities of moral correctness.
    """
    def __init__(self, actions, features, alpha=0.3, temperature=0.15):
        if len(set(actions)) != len(actions) or not actions or len(set(features)) != len(features):
            raise ValueError('Unique actions and features are required.')
        if not finite(alpha) or not 0 < alpha <= 1 or not finite(temperature) or temperature <= 0:
            raise ValueError('Invalid learning rate or temperature.')
        self.actions, self.features = tuple(actions), tuple(features)
        self.alpha, self.temperature = alpha, temperature
        self.terms = ['bias'] + list(features) + [
            left + '*' + right for i, left in enumerate(features) for right in features[i + 1:]]
        self.weights = {action: [0.5] + [0.0] * (len(self.terms) - 1) for action in actions}
        self.revision = 0

    def vector(self, context):
        if set(context) != set(self.features) or any(not finite(v) or not -1 <= v <= 1 for v in context.values()):
            raise ValueError('Context must match the versioned feature schema with finite values in [-1,1].')
        values = [context[name] for name in self.features]
        return [1.0] + values + [left * right for i, left in enumerate(values) for right in values[i + 1:]]

    def values(self, context):
        vector = self.vector(context)
        return {action: max(0.0, min(1.0, sum(w * x for w, x in zip(self.weights[action], vector))))
                for action in self.actions}

    def explain(self, context):
        """Attribute value estimates to weights; this is not a moral warrant."""
        vector = self.vector(context)
        values = self.values(context)
        return {action: {'value': values[action],
                         'raw_value': sum(w * x for w, x in zip(self.weights[action], vector)),
                         'terms': [{'term': term, 'feature_value': x, 'weight': w, 'contribution': w * x}
                                   for term, x, w in zip(self.terms, vector, self.weights[action])]}
                for action in self.actions}

    def distribution(self, values):
        if set(values) != set(self.actions) or any(not finite(v) for v in values.values()):
            raise ValueError('A finite value is required for every action.')
        peak = max(values.values())
        terms = {a: math.exp((v - peak) / self.temperature) for a, v in values.items()}
        return {a: v / sum(terms.values()) for a, v in terms.items()}

    def choose(self, values, rng, epsilon=0.0):
        if not finite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError('Invalid exploration probability.')
        probabilities = {a: (1 - epsilon) * p + epsilon / len(self.actions)
                         for a, p in self.distribution(values).items()}
        draw, cumulative = rng.random(), 0.0
        for action, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                return action, probabilities
        return self.actions[-1], probabilities

    def update(self, context, action, outcome_score):
        if action not in self.weights or not finite(outcome_score) or not 0 <= outcome_score <= 1:
            raise ValueError('Only a finite observed outcome score for an available action may train the policy.')
        vector = self.vector(context)
        predicted = sum(w * x for w, x in zip(self.weights[action], vector))
        correction = self.alpha * (outcome_score - predicted) / sum(x * x for x in vector)
        self.weights[action] = [w + correction * x for w, x in zip(self.weights[action], vector)]
        self.revision += 1

    def snapshot(self):
        return {'version': VERSION, 'actions': list(self.actions), 'features': list(self.features),
                'terms': self.terms, 'alpha': self.alpha, 'temperature': self.temperature,
                'weights': deepcopy(self.weights), 'revision': self.revision}

    @classmethod
    def restore(cls, saved):
        if saved.get('version') != VERSION:
            raise ValueError('Unsupported policy version.')
        policy = cls(saved['actions'], saved['features'], saved['alpha'], saved['temperature'])
        if policy.terms != saved['terms'] or set(saved['weights']) != set(policy.actions):
            raise ValueError('Policy feature/action schema mismatch.')
        if any(len(row) != len(policy.terms) or any(not finite(v) for v in row)
               for row in saved['weights'].values()):
            raise ValueError('Invalid policy weights.')
        if type(saved.get('revision')) is not int or saved['revision'] < 0:
            raise ValueError('Invalid policy revision.')
        policy.weights, policy.revision = deepcopy(saved['weights']), saved['revision']
        return policy


class ExperienceStore:
    """Append-only observed experiences and model checkpoints, isolated by arm/seed.

    Counterfactual evaluator rewards never enter retrieval. Held-out observations
    can be inspected in the journal but cannot train or be retrieved as precedent.
    """
    def __init__(self, path, *, retain_memory=True):
        self.retain_memory = retain_memory
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY, arm TEXT NOT NULL, seed INTEGER NOT NULL,
                phase TEXT NOT NULL CHECK(phase IN ('train','validation','test')),
                step INTEGER NOT NULL, case_id TEXT NOT NULL, group_id TEXT NOT NULL,
                record TEXT NOT NULL, learned INTEGER NOT NULL,
                UNIQUE(arm,seed,phase,step));
            CREATE TABLE IF NOT EXISTS checkpoints (
                arm TEXT NOT NULL, seed INTEGER NOT NULL, phase TEXT NOT NULL,
                policy TEXT NOT NULL, PRIMARY KEY(arm,seed,phase));
            CREATE INDEX IF NOT EXISTS training_memory ON experiences(arm,seed,phase);
            CREATE TABLE IF NOT EXISTS memory (
                experience_id INTEGER PRIMARY KEY REFERENCES experiences(id),
                arm TEXT NOT NULL, seed INTEGER NOT NULL, action TEXT NOT NULL,
                schema TEXT NOT NULL, features TEXT NOT NULL, score REAL NOT NULL,
                case_id TEXT NOT NULL, group_id TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS memory_lookup ON memory(arm,seed,action,schema);
        ''')
        found = self.db.execute("SELECT value FROM metadata WHERE key='version'").fetchone()
        if found and found[0] != VERSION:
            self.db.close()
            raise ValueError('Unsupported experience database version.')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('version',?)", (VERSION,))

    def close(self):
        self.db.close()

    def record(self, *, arm, seed, phase, step, case, action, values, probabilities,
               feedback, retrieval, policy, learn=False):
        if phase != case['split'] or (learn and phase != 'train'):
            raise ValueError('Held-out cases cannot train; phase must match the frozen case split.')
        if action not in policy.actions or action not in case['candidate_actions']:
            raise ValueError('Unavailable action.')
        score = feedback['outcome_score']
        if not finite(score) or not 0 <= score <= 1 or not feedback.get('evidence'):
            raise ValueError('Feedback requires a finite outcome score and recorded evidence.')
        before = policy.snapshot()
        try:
            with self.db:
                if learn:
                    policy.update(case['features'], action, score)
                record = {'policy_version': VERSION, 'feature_schema': case['feature_schema'],
                          'case_id': case['id'], 'group_id': case['group_id'],
                          'features': case['features'], 'candidate_actions': case['candidate_actions'],
                          'selected_action': action, 'values': values, 'selection_probabilities': probabilities,
                          'feedback': feedback, 'retrieval': retrieval,
                          'policy_before': before, 'policy_after': policy.snapshot()}
                cursor = self.db.execute('INSERT INTO experiences(arm,seed,phase,step,case_id,group_id,record,learned) '
                                         'VALUES (?,?,?,?,?,?,?,?)',
                                         (arm, seed, phase, step, case['id'], case['group_id'], json.dumps(record), int(learn)))
                if phase == 'train' and self.retain_memory:
                    self.db.execute('INSERT INTO memory VALUES (?,?,?,?,?,?,?,?,?)',
                                    (cursor.lastrowid, arm, seed, action, case['feature_schema'],
                                     json.dumps(case['features']), score, case['id'], case['group_id']))
        except Exception:
            restored = ContextualPolicy.restore(before)
            policy.weights, policy.revision = restored.weights, restored.revision
            raise

    def checkpoint(self, arm, seed, phase, policy):
        with self.db:
            self.db.execute('INSERT INTO checkpoints VALUES (?,?,?,?)',
                            (arm, seed, phase, json.dumps(policy.snapshot())))

    def retrieve(self, arm, seed, context, action, feature_schema, limit=12):
        found = []
        for row in self.db.execute('SELECT * FROM memory WHERE arm=? AND seed=? AND action=? AND schema=?',
                                   (arm, seed, action, feature_schema)):
            features = json.loads(row['features'])
            if set(features) != set(context):
                continue
            distance = sum((features[k] - context[k]) ** 2 for k in context) / len(context)
            found.append({'experience_id': row['experience_id'], 'case_id': row['case_id'],
                          'group_id': row['group_id'], 'distance': distance,
                          'outcome_score': row['score']})
        # Choose neighbors by features and recency, never by whether feedback is
        # favorable. Contrasts are highlighted separately, not balanced into a
        # misleading 50:50 reward estimate.
        return sorted(found, key=lambda row: (row['distance'], -row['experience_id']))[:limit]


def values_with_memory(policy, context, store, arm, seed, schema, blend=0.25):
    if not finite(blend) or not 0 <= blend <= 1:
        raise ValueError('Invalid memory blend.')
    values, audit = policy.values(context), {}
    for action in policy.actions:
        neighbors = store.retrieve(arm, seed, context, action, schema)
        if not neighbors:
            continue
        weighted = [(math.exp(-row['distance'] / 0.25), row) for row in neighbors]
        estimate = sum(w * row['outcome_score'] for w, row in weighted) / sum(w for w, _ in weighted)
        strength = blend * math.exp(-neighbors[0]['distance'] / 0.25)
        values[action] = (1 - strength) * values[action] + strength * estimate
        audit[action] = {'neighbors': neighbors, 'memory_estimate': estimate, 'blend': strength,
                         'success_example': next((r['experience_id'] for r in neighbors if r['outcome_score'] == 1), None),
                         'failure_example': next((r['experience_id'] for r in neighbors if r['outcome_score'] == 0), None)}
    return values, audit
