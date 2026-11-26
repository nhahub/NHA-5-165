import React from 'react';
import { useTheme } from '../context/ThemeContext';

export default function PredictionCard({ prediction, engineeredFeatures }) {
  const { isDark, colors } = useTheme();

  if (!prediction) {
    return (
      <div style={{
        backgroundColor: colors.bgCard,
        borderRadius: '16px',
        padding: '36px',
        textAlign: 'center',
        border: `2px dashed ${colors.border}`,
        color: colors.textSecondary
      }}>
        <div style={{ fontSize: '42px', marginBottom: '12px' }}>⚡</div>
        <h3 style={{ fontSize: '15px', fontWeight: '700', color: colors.textPrimary }}>
          Awaiting Simulation Trigger
        </h3>
        <p style={{ fontSize: '13px', marginTop: '4px' }}>
          Adjust building parameters on the left and click <strong>"Compute Thermal Loads"</strong> to predict heating and cooling requirements.
        </p>
      </div>
    );
  }

  const { heating_load_kwh, cooling_load_kwh, total_load_kwh, algorithm_used, inference_source } = prediction;
  const isAws = inference_source === 'aws-sagemaker';

  return (
    <div style={{
      backgroundColor: colors.bgCard,
      borderRadius: '16px',
      padding: '24px',
      boxShadow: isDark ? '0 8px 32px rgba(0,0,0,0.4)' : '0 4px 20px rgba(0,0,0,0.06)',
      border: `1px solid ${colors.border}`,
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    }}>
      {/* Card Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '16px', fontWeight: '800', color: colors.textPrimary, margin: 0 }}>
            Inference Results
          </h2>
          <p style={{ fontSize: '12px', color: colors.textSecondary, marginTop: '2px' }}>
            Simulated annual building conditioning demand per square meter
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {isAws && (
            <span style={{
              fontSize: '11px',
              fontWeight: '700',
              backgroundColor: isDark ? 'rgba(245, 158, 11, 0.2)' : '#fffbeb',
              color: '#d97706',
              padding: '4px 12px',
              borderRadius: '20px',
              border: '1px solid rgba(245, 158, 11, 0.45)'
            }}>
              ☁️ AWS SageMaker
            </span>
          )}
          <span style={{
            fontSize: '11px',
            fontWeight: '700',
            backgroundColor: isDark ? 'rgba(59, 130, 246, 0.2)' : '#eff6ff',
            color: '#3b82f6',
            padding: '4px 12px',
            borderRadius: '20px',
            border: '1px solid rgba(59, 130, 246, 0.4)'
          }}>
            {algorithm_used}
          </span>
        </div>
      </div>

      {/* 3 Metric Cards with Radiant Gradients */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
        {/* Heating Load Card */}
        <div style={{
          background: isDark
            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(249, 115, 22, 0.08) 100%)'
            : 'linear-gradient(135deg, #fff1f2 0%, #fff7ed 100%)',
          border: isDark ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid #fecdd3',
          borderRadius: '14px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', fontWeight: '700', color: '#f43f5e', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Heating Load (Y₁)
            </span>
            <span style={{ fontSize: '20px' }}>🔥</span>
          </div>

          <div style={{ fontSize: '32px', fontWeight: '800', color: isDark ? '#fca5a5' : '#be123c', marginTop: '10px', fontFamily: 'monospace' }}>
            {heating_load_kwh}
          </div>
          <span style={{ fontSize: '12px', color: colors.textSecondary, marginTop: '2px' }}>
            kWh/m² &bull; Annual thermal demand
          </span>
        </div>

        {/* Cooling Load Card */}
        <div style={{
          background: isDark
            ? 'linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.08) 100%)'
            : 'linear-gradient(135deg, #ecfeff 0%, #eff6ff 100%)',
          border: isDark ? '1px solid rgba(6, 182, 212, 0.3)' : '1px solid #bae6fd',
          borderRadius: '14px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', fontWeight: '700', color: '#0284c7', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Cooling Load (Y₂)
            </span>
            <span style={{ fontSize: '20px' }}>❄️</span>
          </div>

          <div style={{ fontSize: '32px', fontWeight: '800', color: isDark ? '#7dd3fc' : '#0369a1', marginTop: '10px', fontFamily: 'monospace' }}>
            {cooling_load_kwh}
          </div>
          <span style={{ fontSize: '12px', color: colors.textSecondary, marginTop: '2px' }}>
            kWh/m² &bull; Annual cooling load
          </span>
        </div>

        {/* Total Load Card */}
        <div style={{
          background: isDark
            ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(139, 92, 246, 0.1) 100%)'
            : 'linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%)',
          border: isDark ? '1px solid rgba(99, 102, 241, 0.35)' : '1px solid #c7d2fe',
          borderRadius: '14px',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', fontWeight: '700', color: '#6366f1', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Combined Intensity
            </span>
            <span style={{ fontSize: '20px' }}>⚡</span>
          </div>

          <div style={{ fontSize: '32px', fontWeight: '800', color: isDark ? '#a5b4fc' : '#4338ca', marginTop: '10px', fontFamily: 'monospace' }}>
            {total_load_kwh}
          </div>
          <span style={{ fontSize: '12px', color: colors.textSecondary, marginTop: '2px' }}>
            kWh/m² &bull; Total building energy
          </span>
        </div>
      </div>

      {/* Pipeline Feature Calculations */}
      {engineeredFeatures && (
        <div style={{
          backgroundColor: colors.bgSubtle,
          borderRadius: '10px',
          padding: '12px 18px',
          border: `1px solid ${colors.borderSubtle}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          fontSize: '12px',
          color: colors.textSecondary
        }}>
          <div>
            <strong>Wall/Surface Ratio:</strong> <span style={{ color: colors.textPrimary, fontFamily: 'monospace' }}>{engineeredFeatures.Wall_Surface_Ratio}</span>
          </div>
          <div>
            <strong>Glazing &times; Height:</strong> <span style={{ color: colors.textPrimary, fontFamily: 'monospace' }}>{engineeredFeatures.Glazing_x_Height} m</span>
          </div>
          <span style={{ fontSize: '11px', color: colors.textMuted }}>
            Calculated on-the-fly via <code>FunctionTransformer</code>
          </span>
        </div>
      )}
    </div>
  );
}
