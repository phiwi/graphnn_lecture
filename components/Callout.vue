<template>
  <div
    class="callout"
    :class="[`callout-${typeClass}`]"
    role="note"
    :aria-label="computedTitle"
    :style="styleVars"
  >
    <div v-if="computedTitle" class="callout-title">{{ computedTitle }}</div>
    <div class="callout-body">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

type CalloutType = 'tip' | 'info' | 'warning' | 'danger'

const props = withDefaults(
  defineProps<{ type?: CalloutType; title?: string }>(),
  {
    type: 'info',
    title: '',
  },
)

const typeClass = computed<CalloutType>(() => props.type ?? 'info')

const styleVars = computed(() => {
  const palette: Record<CalloutType, { accent: string; background: string }> = {
    tip: { accent: '#0891b2', background: '#ecfeff' },
    info: { accent: '#2563eb', background: '#eff6ff' },
    warning: { accent: '#f59e0b', background: '#fffbeb' },
    danger: { accent: '#dc2626', background: '#fef2f2' },
  }

  const selected = palette[typeClass.value]

  return {
    '--callout-accent': selected.accent,
    '--callout-bg': selected.background,
  }
})

const computedTitle = computed(() => {
  if (props.title) {
    return props.title
  }
  const labelMap: Record<CalloutType, string> = {
    tip: 'Tip',
    info: 'Info',
    warning: 'Warning',
    danger: 'Caution',
  }
  return labelMap[typeClass.value]
})
</script>

<style scoped>
.callout {
  border-radius: 0.5rem;
  padding: 0.85rem 1.1rem;
  margin: 1.2rem 0;
  border-left: 0.35rem solid var(--callout-accent, #2563eb);
  background: var(--callout-bg, #f1f5f9);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
  color: #0f172a;
}

.callout-title {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
  color: var(--callout-accent, #2563eb);
}

.callout-body :first-child {
  margin-top: 0;
}

.callout-body :last-child {
  margin-bottom: 0;
}
</style>
