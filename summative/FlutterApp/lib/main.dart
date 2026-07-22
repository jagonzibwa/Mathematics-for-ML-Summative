import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Public API base URL (Render). For local testing use:
///   Android emulator: `http://10.0.2.2:8000`
///   Physical device:  `http://your-pc-lan-ip:8000`
const String apiBaseUrl = 'https://studentgradeapi.onrender.com';

void main() => runApp(const GradePredictorApp());

class GradePredictorApp extends StatelessWidget {
  const GradePredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Student Grade Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1A5CAB)),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

/// Describes one model input so the form can be built from a single list.
class FieldSpec {
  const FieldSpec(this.key, this.label, this.hint, {this.isInt = true});

  final String key;
  final String label;
  final String hint;
  final bool isInt;
}

const List<FieldSpec> fieldSpecs = [
  FieldSpec('G1', 'First-period grade (G1)', '0 – 20', isInt: false),
  FieldSpec('G2', 'Second-period grade (G2)', '0 – 20', isInt: false),
  FieldSpec('failures', 'Past class failures', '0 – 4'),
  FieldSpec('Medu', "Mother's education level", '0 (none) – 4 (higher ed.)'),
  FieldSpec('age', 'Age', '15 – 22'),
  FieldSpec('goout', 'Going out with friends', '1 (very low) – 5 (very high)'),
  FieldSpec('studytime', 'Weekly study time', '1 (<2h) – 4 (>10h)'),
  FieldSpec('absences', 'School absences', '0 – 93'),
];

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final f in fieldSpecs) f.key: TextEditingController(),
  };

  bool _loading = false;
  String? _error;
  double? _predictedGrade;
  String? _modelUsed;
  String? _interpretation;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _predict() async {
    setState(() {
      _loading = true;
      _error = null;
      _predictedGrade = null;
    });

    // Client-side check: every field must be filled with a number.
    final body = <String, num>{};
    for (final f in fieldSpecs) {
      final text = _controllers[f.key]!.text.trim();
      final value = num.tryParse(text);
      if (text.isEmpty || value == null) {
        setState(() {
          _loading = false;
          _error = text.isEmpty
              ? 'Missing value: please fill in "${f.label}".'
              : '"${f.label}" must be a number.';
        });
        return;
      }
      body[f.key] = f.isInt ? value.toInt() : value;
    }

    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 60));

      final data = jsonDecode(response.body);
      if (response.statusCode == 200) {
        setState(() {
          _predictedGrade = (data['predicted_final_grade'] as num).toDouble();
          _modelUsed = data['model_used'] as String?;
          _interpretation = data['interpretation'] as String?;
        });
      } else if (response.statusCode == 422) {
        // FastAPI / Pydantic validation errors (e.g. out-of-range values).
        final details = (data['detail'] as List)
            .map((e) => '${(e['loc'] as List).last}: ${e['msg']}')
            .join('\n');
        setState(() => _error = 'Invalid input:\n$details');
      } else {
        setState(() => _error = 'Server error (${response.statusCode}).');
      }
    } catch (e) {
      setState(
        () => _error =
            'Could not reach the prediction API. Check your connection.',
      );
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Student Grade Predictor'),
        backgroundColor: theme.colorScheme.primary,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Predict a student\'s final Mathematics grade (0–20) from their '
                'academic history and study habits.',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: Colors.grey[700],
                ),
              ),
              const SizedBox(height: 16),
              for (final f in fieldSpecs) ...[
                TextField(
                  controller: _controllers[f.key],
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  decoration: InputDecoration(
                    labelText: f.label,
                    helperText: 'Range: ${f.hint}',
                    border: const OutlineInputBorder(),
                    isDense: true,
                  ),
                ),
                const SizedBox(height: 14),
              ],
              const SizedBox(height: 4),
              FilledButton(
                onPressed: _loading ? null : _predict,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  textStyle: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                child: _loading
                    ? const SizedBox(
                        height: 22,
                        width: 22,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.5,
                          color: Colors.white,
                        ),
                      )
                    : const Text('Predict'),
              ),
              const SizedBox(height: 16),
              if (_error != null)
                Card(
                  color: theme.colorScheme.errorContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.error_outline,
                          color: theme.colorScheme.error,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _error!,
                            style: TextStyle(
                              color: theme.colorScheme.onErrorContainer,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              if (_predictedGrade != null)
                Card(
                  color: theme.colorScheme.primaryContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Text(
                          'Predicted Final Grade',
                          style: theme.textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${_predictedGrade!.toStringAsFixed(2)} / 20',
                          style: theme.textTheme.displaySmall?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: theme.colorScheme.primary,
                          ),
                        ),
                        if (_interpretation != null) ...[
                          const SizedBox(height: 8),
                          Text(_interpretation!, textAlign: TextAlign.center),
                        ],
                        if (_modelUsed != null) ...[
                          const SizedBox(height: 4),
                          Text(
                            'Model: $_modelUsed',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: Colors.grey[700],
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
