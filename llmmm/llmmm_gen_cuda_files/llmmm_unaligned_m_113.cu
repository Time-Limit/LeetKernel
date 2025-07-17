#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_113_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<113> mm_instantiator;

public:
  UnalignedM_113_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_113_MMInstantiatorWrapper__;

}  // namespace LLMMM
