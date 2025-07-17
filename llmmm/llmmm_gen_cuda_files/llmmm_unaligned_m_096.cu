#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_96_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<96> mm_instantiator;

public:
  UnalignedM_96_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_96_MMInstantiatorWrapper__;

}  // namespace LLMMM
