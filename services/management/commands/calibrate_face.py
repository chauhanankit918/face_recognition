"""Report the cosine-similarity distribution for a labelled set of face photos.

Layout the images one directory per person:

    photos/
      alice/  a1.jpg  a2.jpg  a3.jpg
      bob/    b1.jpg  b2.jpg

Every within-person pair is a genuine pair, every cross-person pair an impostor
pair. The command prints both distributions and suggests values for
FACE_COS_FLOOR / FACE_COS_THRESHOLD / FACE_COS_CEILING.
"""
import itertools
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from services import face_engine

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class Command(BaseCommand):
    help = 'Calibrate face-similarity thresholds against a labelled photo set.'

    def add_arguments(self, parser):
        parser.add_argument('directory', help='Directory of per-person subfolders.')

    def handle(self, *args, **options):
        root = Path(options['directory']).expanduser()
        if not root.is_dir():
            raise CommandError(f'{root} is not a directory.')

        people = self._load_embeddings(root)
        if len(people) < 2:
            raise CommandError(
                'Need at least two people with usable photos to calibrate.'
            )

        genuine = [
            face_engine.cosine_similarity(a, b)
            for embeddings in people.values()
            for a, b in itertools.combinations(embeddings, 2)
        ]
        impostor = [
            face_engine.cosine_similarity(a, b)
            for (_, xs), (_, ys) in itertools.combinations(people.items(), 2)
            for a in xs
            for b in ys
        ]

        self._report('Genuine (same person)', genuine)
        self._report('Impostor (different people)', impostor)

        if not genuine or not impostor:
            self.stdout.write(self.style.WARNING(
                '\nNeed both genuine and impostor pairs to suggest anchors. '
                'Add at least two photos per person.'
            ))
            return

        self._suggest(genuine, impostor)

    def _load_embeddings(self, root):
        people = {}
        for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            embeddings = []
            for path in sorted(person_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    embedding, _ = face_engine.get_embedding(path.read_bytes())
                except face_engine.FaceError as exc:
                    self.stdout.write(self.style.WARNING(
                        f'  skipped {path.name}: {exc.message}'
                    ))
                    continue
                embeddings.append(embedding)
            if embeddings:
                people[person_dir.name] = embeddings
                self.stdout.write(f'{person_dir.name}: {len(embeddings)} photo(s)')
        return people

    def _report(self, label, scores):
        self.stdout.write(self.style.MIGRATE_HEADING(f'\n{label}: {len(scores)} pairs'))
        if not scores:
            self.stdout.write('  (none)')
            return
        scores = np.array(scores)
        for name, value in [
            ('min', scores.min()),
            ('p5', np.percentile(scores, 5)),
            ('median', np.median(scores)),
            ('p95', np.percentile(scores, 95)),
            ('max', scores.max()),
        ]:
            self.stdout.write(f'  {name:>6}: {value:.4f}')

    def _suggest(self, genuine, impostor):
        genuine, impostor = np.array(genuine), np.array(impostor)
        floor = float(np.percentile(impostor, 50))
        threshold = float((np.percentile(impostor, 99) + np.percentile(genuine, 1)) / 2)
        ceiling = float(np.percentile(genuine, 90))

        overlap = (genuine < np.percentile(impostor, 99)).mean()
        self.stdout.write(self.style.MIGRATE_HEADING('\nSuggested settings'))
        self.stdout.write(f'FACE_COS_FLOOR = {floor:.2f}')
        self.stdout.write(f'FACE_COS_THRESHOLD = {threshold:.2f}')
        self.stdout.write(f'FACE_COS_CEILING = {max(ceiling, threshold + 0.1):.2f}')
        if overlap > 0.01:
            self.stdout.write(self.style.WARNING(
                f'\n{overlap:.1%} of genuine pairs fall inside the impostor range; '
                'the two distributions overlap, so no threshold separates them '
                'cleanly. Consider higher-quality enrolment photos.'
            ))
