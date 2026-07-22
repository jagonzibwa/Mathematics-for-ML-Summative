import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_app/main.dart';

void main() {
  testWidgets('Prediction page renders its form and button', (tester) async {
    await tester.pumpWidget(const GradePredictorApp());

    expect(find.text('Predict'), findsOneWidget);
    expect(find.byType(TextField), findsNWidgets(fieldSpecs.length));
  });
}
